
static char help[] = "Bratu nonlinear BVP in 1d.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
       <parameter> = Bratu parameter (0 <= par <= 3.5)\n\n";
/* ------------------------------------------------------------------------

   The Bratu 1D BVP is given by

            - u_xx - lambda*exp(u) = 0,  0 < x < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1.

    A finite difference approximation with the usual 3-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

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
extern PetscErrorCode FormInitialGuess(DM,AppCtx*,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar*,PetscScalar*,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar*,Mat,Mat,AppCtx*);

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
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Bratu 1D BVP problem options","");CHKERRQ(ierr);
  {
    user.param = 1.0;
    ierr       = PetscOptionsReal("-par","Mass parameter","",user.param, &(user.param),NULL);CHKERRQ(ierr);
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
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
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
  PetscReal      lambda,hx;
  PetscScalar    *x;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  lambda = user->param;
  hx     = 1.0/(PetscReal)(Mx-1);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs   - starting grid index (no ghost points)
       xm   - widths of local grid (no ghost points)

  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  /* Compute initial guess over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    if (i == 0 || i == Mx-1) {
      /* boundary conditions are all zero Dirichlet */
      x[i] = 0.0;
    } else {
      x[i] = PetscSinReal(PETSC_PI*i*hx);
    }
  }
  /* Restore vector */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
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
  PetscReal      lambda,hx,sc;
  PetscScalar    u,ue,uw,uxx;

  PetscFunctionBeginUser;
  lambda = user->param;
  hx     = 1.0/(PetscReal)(info->mx-1);
  sc     = hx*hx*lambda;
  /* Compute function over the locally owned part of the grid */
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0 || i == info->mx-1) {
      f[i] = 2.0*x[i];
    } else {
      u  = x[i];
      uw = x[i-1];
      ue = x[i+1];

      if (i-1 == 0) uw = 0.;
      if (i+1 == info->mx-1) ue = 0.;

      uxx  = (2.0*u - uw - ue);
      f[i] = uxx - sc*PetscExpScalar(u);
    }
  }
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
  PetscScalar    lambda,v[3],hx,sc;
  DM             coordDA;
  Vec            coordinates;
  PetscScalar   *coords;

  PetscFunctionBeginUser;
  lambda = user->param;
  /* Extract coordinates */
  ierr = DMGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  hx   = info->xm > 1 ? PetscRealPart(coords[info->xs+1]) - PetscRealPart(coords[info->xs]) : 1.0;
  ierr = DMDAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  sc     = hx*hx*lambda;

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
    row.i = i;
    /* boundary points */
    if (i == 0 || i == info->mx-1) {
      v[0] =  2.0;
      ierr = MatSetValuesStencil(jacpre,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      /* interior grid points */
      k = 0;
      /* west point */
      if (i-1 != 0) {
	v[k]     = -1.0;
	col[k].i = i-1;
	k++;
      }
      /* current grid point */
      v[k] = 2.0 - sc*PetscExpScalar(x[i]);
      col[k].i = row.i;
      k++;
      /* east point */
      if (i+1 != info->mx-1) {
	v[k]     = -1.0;
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
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
