
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

runex8:
	./ex8 -monitor_error -ts_atol 1e-2 -ts_rtol 1e-2 -ts_exact_final_time interpolate -ts_type arkimex

runex8_2:
	./ex8 -ts_atol 1e-2 -ts_rtol 1e-2 -ts_final_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 0

runex8_3:
	./ex8 -ts_atol 1e-2 -ts_rtol 1e-2 -ts_final_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 1
