#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _HHHmod_reg();
extern void _NMDA_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," HHHmod.mod");
fprintf(stderr," NMDA.mod");
fprintf(stderr, "\n");
    }
_HHHmod_reg();
_NMDA_reg();
}
