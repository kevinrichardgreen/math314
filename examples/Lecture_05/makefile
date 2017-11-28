
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratu_1d: bratu_1d.o  chkopts
	-${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	${RM} $<

bratu_1d_nonuniform: bratu_1d_nonuniform.o  chkopts
	-${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	${RM} $<

bvp1: bvp1.o  chkopts
	-${CLINKER} -o $@ $< ${PETSC_SNES_LIB}
	${RM} $<
