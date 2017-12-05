
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_TS_LIB}
	${RM} ex19.o
