
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

fdo1: fdo1.o  chkopts
	-${CLINKER} -o $@ $< ${PETSC_TS_LIB}
	${RM} $<

fdo2: fdo2.o  chkopts
	-${CLINKER} -o $@ $< ${PETSC_TS_LIB}
	${RM} $<
