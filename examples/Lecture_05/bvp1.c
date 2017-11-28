
static char help[] = "2nd order linear BVP in 1d.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's scaling\n\
       <parameter> = epsilon parameter (0 < par )\n\n";
/* ------------------------------------------------------------------------

   The Bratu 1D BVP is given by

             \epsilon u'' +(1+\epsilon) u' + u = 0

    with boundary conditions

             u(0) = 0,  u(1) = 1.

    A finite difference approximation with  3-point stencils
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

    This version can make use of a nonuniform mesh.

  ------------------------------------------------------------------------- */

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

typedef struct {
  PetscReal param;          /* problem parameter */
} AppCtx;

/*
   User-defined routines
*/
extern PetscScalar CoordinateTransformFunction(PetscScalar x);
extern PetscErrorCode SetCoordinates1d(DM da);
extern PetscErrorCode FormInitialGuess(DM,AppCtx*,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar*,PetscScalar*,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar*,Mat,Mat,AppCtx*);
extern PetscErrorCode MySNESPrintSolution(SNES snes);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                         /* nonlinear solver */
  Vec            x;                            /* solution vector */
  AppCtx         user;                         /* user-defined work context */
  PetscInt       its;                          /* iterations for convergence */
  PetscErrorCode ierr;
  DM             da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"2nd order 1D BVP problem options","");CHKERRQ(ierr);
  {
    user.param = 0.1;
    ierr       = PetscOptionsReal("-par","scaling parameter epsilon","",user.param, &(user.param),NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage (parallel) grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,-4,1,1,NULL,&da);CHKERRQ(ierr);
  ierr = SetCoordinates1d(da);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Use the info from the set up DMDA to create global vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function and Jacobian evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Initial approximation to the solution */
  ierr = FormInitialGuess(da,&user,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  ierr = MySNESPrintSolution(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CoordinateTransformFunction"
/*
  monotonic increasing function to transform grid coordinates
  f: [0,1] -> [0,1]
*/
PetscScalar CoordinateTransformFunction(PetscScalar x)
{
  /* return x; */
  /* return PetscPowScalar(x,2.0); */
  return PetscPowScalar(x,3.0);
}
#undef __FUNCT__
#define __FUNCT__ "SetCoordinates1d"
/*
  Set the coordinates of a 1D DMDA
*/
PetscErrorCode SetCoordinates1d(DM da)
{
  PetscErrorCode ierr;
  PetscInt       i,start,m;
  Vec            local,global;
  PetscScalar    *coorslocal;
  DM             cda;

  PetscFunctionBeginUser;
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&global);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&start,0,0,&m,0,0);CHKERRQ(ierr);
  for (i=start; i<start+m; i++) {
    coorslocal[i] = CoordinateTransformFunction(coorslocal[i]);
  }
  ierr = DMDAVecRestoreArray(cda,local,&coorslocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(cda,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(cda,local,INSERT_VALUES,global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   da - The DM
   user - user-defined application context

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(DM da,AppCtx *user,Vec X)
{
  PetscInt       i,Mx,xs,xm;
  PetscErrorCode ierr;
  DM             cda;
  Vec            local;
  PetscScalar   *coors;
  PetscScalar   *x;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,local,&coors);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs   - starting grid index (no ghost points)
       xm   - widths of local grid (no ghost points)

  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  /* Compute initial guess over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    if (i == 0) {
      x[i] = 0.0;
    } else if (i == Mx-1) {
      x[i] = 1.0;
    } else {
      /* x[i] = PetscExpScalar(-coors[i])*PetscExpScalar(1.0); */
      x[i] = coors[i];
    }
  }
  /* Restore vector */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArrayRead(cda,local,&coors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch
 */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar *x,PetscScalar *f,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      epsilon,he,hw;
  PetscScalar    u,ue,uw,uxx,ux,factor;
  DM             cda;
  Vec            local;
  PetscScalar   *coors;

  PetscFunctionBeginUser;
  epsilon = user->param;

  /* Get coordinates */
  ierr = DMGetCoordinateDM(info->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(info->da,&local);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,local,&coors);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info->xs; i<info->xs+info->xm; i++) {
    /* Handle BC's */
    if (i == 0) {
      f[i] = x[i];
    } else if ( i == info->mx-1) {
      f[i] = x[i]-1.0;
    } else {
      /* determine east and west step sizes */
      hw = coors[i] - coors[i-1];
      he = coors[i+1] - coors[i];
      factor = 1.0/(he*hw*hw + hw*he*he);

      /* compute the second derivative term */
      u  = x[i];
      uw = x[i-1];
      ue = x[i+1];
      if (i-1 == 0) uw = 0.;
      if (i+1 == info->mx-1) ue = 1.;
      uxx  = 2.0*factor*((he+hw)*u - he*uw - hw*ue);

      /* compute the first derivative term */
      ux = factor*((hw*hw-he*he)*u - hw*hw*ue + he*he*uw);

      /* evaluate the residual */
      f[i] = epsilon*uxx + (1+epsilon)*ux - u;
    }
  }

  ierr = DMDAVecRestoreArrayRead(cda,local,&coors);CHKERRQ(ierr);

  ierr = PetscLogFlops(11.0*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
   FormJacobianLocal - Evaluates Jacobian matrix on local process patch
*/
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar *x,Mat jac,Mat jacpre,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  MatStencil     col[5],row;
  PetscScalar    epsilon,v[3],he,hw,factor;
  DM             coordDA;
  Vec            coordinates;
  PetscScalar   *coors;

  PetscFunctionBeginUser;
  epsilon = user->param;
  /* Extract coordinates */
  ierr = DMGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(coordDA, coordinates, &coors);CHKERRQ(ierr);
  /*
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (i=info->xs; i<info->xs+info->xm; i++) {
    /* determine east and west step sizes */
    hw = coors[i] - coors[i-1];
    he = coors[i+1] - coors[i];
    factor = 1.0/(he*hw*hw+hw*he*he);

    row.i = i;
    /* boundary points */
    if (i == 0 ) {
      v[0] =  1.0; /* NOTE: either hw or he are undefined at BC */
      ierr = MatSetValuesStencil(jacpre,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == info->mx-1) {
      v[0] =  1.0; /* NOTE: either hw or he are undefined at BC */
      ierr = MatSetValuesStencil(jacpre,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    }else {
      /* interior grid points */
      k = 0;
      /* west point */
      if (i-1 != 0) {
	v[k]     = (-2.0*he*epsilon+he*he*(1+epsilon))*factor;
	col[k].i = i-1;
	k++;
      }
      /* current grid point */
      v[k] = factor*(2*(he+hw)*epsilon+(hw*hw-he*he)*(1+epsilon)) - x[i];
      col[k].i = row.i;
      k++;
      /* east point */
      if (i+1 != info->mx-1) {
	v[k]     = (-2.0*hw*epsilon-hw*hw*(1+epsilon))*factor;
	col[k].i = i+1;
	k++;
      }
      ierr = MatSetValuesStencil(jacpre,1,&row,k,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jacpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jacpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Restore coordinate array */
  ierr = DMDAVecRestoreArrayRead(coordDA, coordinates, &coors);CHKERRQ(ierr);
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MySNESPrintSolution"
/*
   MySNESPrintSolution - print out results of an SNES solve performed
   with nonuniform 1D grid
*/
PetscErrorCode MySNESPrintSolution(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i,Mx;
  DM             da,coordDA;
  Vec            coordinates, X;
  const PetscScalar   *coors, *x;

  PetscFunctionBeginUser;
  /* Extract coordinates */
  ierr = SNESGetDM(snes, &da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da, &coordDA);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(coordDA, coordinates, &coors);CHKERRQ(ierr);

  /* Extract solution */
  ierr = SNESGetSolution(snes, &X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  for (i=0; i<Mx; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d %16.14f %16.14f\n",i,coors[i],x[i]);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(coordDA, coordinates, &coors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
