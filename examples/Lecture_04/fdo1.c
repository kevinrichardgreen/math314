
static char help[] = "Forced, damped oscillator.\n"
  "Formulated for fully explicit and fully implicit methods.\n\n";

#include <petscts.h>

/*
  Parameters needed for the ODE
*/
typedef struct _Parameters Parameters;
struct _Parameters {
  PetscReal   m,c,k;   // eqn parameters
  PetscScalar F,omega; // forcing parameters
  PetscScalar x0,xp0;  // initial conditions
};
/*
  Forcing function of the RHS
*/
PetscScalar forcing_function(PetscReal time, Parameters* params)
{
  return params->F*sin(params->omega*time);
}
/*
  RHS evaluation
*/
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  Parameters*       params = (Parameters*)(ctx);
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = (-params->k*x[0] - params->c*x[1] + params->c*forcing_function(t,params)) / params->m;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
  RHS Jacobian
*/
#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  Parameters*       params = (Parameters*)(ctx);
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = 0.0;                    J[0][1] = 1.0;
  J[1][0] = -params->k/params->m;   J[1][1] = -params->c/params->m;
  ierr    = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions"
static PetscErrorCode SetInitialConditions(Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  Parameters*       params = (Parameters*)(ctx);
  PetscScalar       *x;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = params->x0;
  x[1] = params->xp0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
  User-defined monitor for solution
*/
#undef __FUNCT__
#define __FUNCT__ "MonitorSolution"
static PetscErrorCode MonitorSolution(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  /* Parameters     params = (Parameters)ctx; */
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  if (step == 0)
    ierr = PetscPrintf(PETSC_COMM_WORLD,"# step time x xp\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4D %12.8e %12.8e %12.8e\n",step,(double)t,(double)x[0],(double)x[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  /* PetscFunctionList plist = NULL; */
  /* char              pname[256]; */
  TS                ts;            /* nonlinear solver */
  Vec               x,r;           /* solution, residual vectors */
  Mat               A;             /* Jacobian matrix */
  Parameters*       params;
  PetscBool         use_monitor;
  PetscInt          steps,maxsteps=1000000,rejects;
  PetscReal         final_time=10.0,ftime;
  PetscErrorCode    ierr;
  PetscMPIInt       size;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  ierr  = PetscMalloc(sizeof(Parameters),&params);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Forced damped oscillator simulation options","");CHKERRQ(ierr);
  {
    use_monitor   = PETSC_FALSE;
    ierr          = PetscOptionsBool("-monitor_solution","Print out numerical solution during simulation","",use_monitor,&use_monitor,NULL);CHKERRQ(ierr);
    params->m     = 1;
    params->c     = 1;
    params->k     = 10;
    ierr          = PetscOptionsReal("-m","Mass parameter","",params->m, &(params->m),NULL);CHKERRQ(ierr);
    ierr          = PetscOptionsReal("-c","Damping parameter","",params->c, &(params->c),NULL);CHKERRQ(ierr);
    ierr          = PetscOptionsReal("-k","Stiffness parameter","",params->k,&(params->k),NULL);CHKERRQ(ierr);
    params->omega = 1;
    params->F     = 1;
    ierr          = PetscOptionsReal("-omega","Forcing frequency","",params->omega,&(params->omega),NULL);CHKERRQ(ierr);
    ierr          = PetscOptionsReal("-F","Forcing amplitude","",params->F,&(params->F),NULL);CHKERRQ(ierr);
    params->x0    = 1;
    params->xp0   = 0;
    ierr          = PetscOptionsReal("-x0","Initial position","",params->x0,&(params->x0),NULL);CHKERRQ(ierr);
    ierr          = PetscOptionsReal("-xp0","Initial velocity","",params->xp0,&(params->xp0),NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,2,2,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,params);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,params);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,final_time);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetMaxStepRejections(ts,10);CHKERRQ(ierr);
  if (use_monitor) {
    ierr = TSMonitorSet(ts,&MonitorSolution,&params,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SetInitialConditions(x,params);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetStepRejections(ts,&rejects);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"# steps %D (%D rejected), ftime %g\n",steps,rejects,(double)ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  /* ierr = VecDestroy(&mon.x);CHKERRQ(ierr); */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
