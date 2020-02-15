#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <ctype.h>
#include <libgen.h>

#include "CMS_config.h"
#include "CMS_basis.h"

// For _mm_malloc and _mm_free
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#endif
#if defined(__GNUC__) || (__clang__)
#include <mm_malloc.h>
#endif

#define ELEN         50
#define SLEN         5
#define MAXNS        3
#define MAXATOMNAME  2
#define A2BOHR       (1.0/0.52917720859)
#define CARTESIAN    0
#define SPHERICAL    1

static char etable[ELEN][MAXATOMNAME + 1] =
{
  "H",  "He", "Li", "Be", "B",
  "C",  "N",  "O",  "F",  "Ne",
  "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca",
  "Sc", "Ti", "V",  "Cr", "Mn",
  "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br",
  "Kr", "Rb", "Sr", "Y",  "Zr",
  "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn"
};

static char mtable[SLEN] = {'S',  'P', 'D', 'F', 'G'};


/* Normalize the shells
 *
 * On output, basis->bx_cc[][] structure will be altered.
 * On output, basis->norm[] structure will be filled.
 * Recognizes shells with zero orbital exponent.
 * Code is a good example of looping through elements of a basis set.
 */
static void normalization(BasisSet_t basis)
{
    double sum;
    double temp;
    double temp2;
    double temp3;
    double xnorm;
    double a1;
    double a2;
    int i;
    int j;
    int k;
    double power;
    int shell;

    if (basis->basistype == SPHERICAL)
    {
        CMS_PRINTF (1, "Warning: performing normalization for SPHERICAL\n");
    }
    else
    {
        CMS_PRINTF (1, "Warning: NOT performing normalization for CARTESIAN\n");
        return;
    }

    for (i = 0; i < basis->bs_nshells; i++)
    {
        sum = 0.0;
        for (j = 0; j < basis->bs_nexp[i]; j++)
        {
            for (k = 0; k <= j; k++)
            {
                a1 = basis->bs_exp[i][j];
                a2 = basis->bs_exp[i][k];
                temp = basis->bs_cc[i][j] * basis->bs_cc[i][k]; 
                temp2 = basis->bs_momentum[i] + 1.5;
                temp3 = 2.0 * sqrt (a1 * a2) / (a1 + a2);
                temp3 = pow (temp3, temp2);
                temp = temp * temp3;
                sum = sum + temp;
                if (j != k)
                {
                    sum = sum + temp;
                }
            }
        }
        xnorm = 1.0 / sqrt (sum);
        shell = basis->bs_momentum[i];
        power = (double) shell *0.5 + 0.75;
        for (j = 0; j < basis->bs_nexp[i]; j++)
        {
            basis->bs_cc[i][j] *= xnorm;
            if (basis->bs_exp[i][j] == 0.0)
            {
                basis->bs_norm[i][j] = 1.0;
            }
            else
            {
                basis->bs_norm[i][j] = pow (basis->bs_exp[i][j], power);
            }
        }              
    }    
}

static inline double vector_min(size_t length, const double vector[restrict static length]) {
    double result = vector[0];
    for (size_t i = 1; i < length; i++) {
        const double element = vector[i];
        result = (element < result) ? element : result;
    }
    return result;
}

CMSStatus_t CMS_createBasisSet(BasisSet_t *_basis)
{
    BasisSet_t basis;
    basis = (BasisSet_t) malloc(sizeof(struct BasisSet));
    CMS_ASSERT(basis != NULL);
    memset(basis, 0, sizeof(struct BasisSet));

    *_basis = basis;
    return CMS_STATUS_SUCCESS;
}

CMSStatus_t CMS_destroyBasisSet(BasisSet_t basis)
{
    free(basis->f_start_id);
    free(basis->f_end_id);

    for (int i = 0; i < basis->bs_nshells; i++)
    {
        ALIGNED_FREE(basis->bs_cc[i]);
        ALIGNED_FREE(basis->bs_exp[i]);
        ALIGNED_FREE(basis->bs_norm[i]);
    }
    free(basis->bs_cc);
    free(basis->bs_exp);
    free(basis->bs_norm);

    if (basis->guess != NULL)
    {
        for (int i = 0; i < basis->bs_natoms; i++) free(basis->guess[i]);
        free(basis->guess);
    }    
    free(basis->eid);
    free(basis->xn);
    free(basis->yn);
    free(basis->zn);   
    free(basis->charge);
    free(basis->bs_eptr);
    free(basis->bs_atom_start);
    free(basis->bs_nexp);
    free(basis->bs_momentum);
    free(basis->cc);
    free(basis->exp);
    free(basis->minexp);
    free(basis->norm);
    
    free(basis);

    return CMS_STATUS_SUCCESS;
}


// Create the data structure of the shells for the given molecule.
CMSStatus_t CMS_parse_molecule(BasisSet_t basis)
{
    int natoms;
    int nshells;   
    int nfunctions;
    int maxdim;
    int max_momentum;
    int max_nexp;
    int max_nexp_id;
    int eid;
    int atom_start;
    int atom_end;

    // count number of shells in basis set for all atoms in the molecule
    natoms = basis->natoms; // number of atoms in the molecule
    nshells = 0;
    for (uint32_t i = 0; i < natoms; i++) {
        eid = basis->eid[i];
        atom_start = basis->bs_atom_start[basis->bs_eptr[eid - 1]];
        atom_end = basis->bs_atom_start[basis->bs_eptr[eid - 1] + 1];
        nshells += atom_end - atom_start;
    }
    
    // start of shell info for each atom
    basis->s_start_id = (uint32_t *)malloc (sizeof(uint32_t) * (natoms + 1));
    // start and end indices of the functions for each shell;
    // this is useful when constructing the Fock matrix
    basis->f_start_id = (uint32_t *)malloc (sizeof(uint32_t) * nshells);
    basis->f_end_id   = (uint32_t *)malloc (sizeof(uint32_t) * nshells);
    // shell centers
    basis->xyz0 = (double *)ALIGNED_MALLOC(sizeof(double) * nshells * 4);
    basis->nexp =     (uint32_t *)malloc(sizeof(uint32_t) * nshells); // number of primitives
    basis->cc =       (double **) malloc(sizeof(double *) * nshells); // contraction coefs
    basis->exp =      (double **) malloc(sizeof(double *) * nshells); // orbital exponents
    basis->minexp =   (double *)  malloc(sizeof(double)   * nshells); // how used?
    basis->norm =     (double **) malloc(sizeof(double *) * nshells); // how used?
    basis->momentum = (uint32_t *)malloc(sizeof(uint32_t) * nshells); //
    CMS_ASSERT(basis->s_start_id != NULL);
    CMS_ASSERT(basis->f_start_id != NULL);
    CMS_ASSERT(basis->f_end_id   != NULL);
    CMS_ASSERT(basis->xyz0       != NULL);
    CMS_ASSERT(basis->nexp       != NULL);
    CMS_ASSERT(basis->cc         != NULL);
    CMS_ASSERT(basis->minexp     != NULL);
    CMS_ASSERT(basis->norm       != NULL);
    CMS_ASSERT(basis->momentum   != NULL);
    basis->nshells = nshells; // number of shells for this molecule
    basis->mem_size += sizeof(uint32_t) * (natoms + 1 + nshells * 4)
                     + sizeof(double) * nshells * 9 + sizeof(double *) * nshells * 3;

    // loop over atoms in the molecule
    nshells = 0;
    nfunctions = 0;
    maxdim = 0;
    max_momentum = 0;
    max_nexp = 0;
    max_nexp_id = 0;
    for (uint32_t i = 0; i < natoms; i++) {
        eid = basis->eid[i];    
        atom_start = basis->bs_atom_start[basis->bs_eptr[eid - 1]];
        atom_end   = basis->bs_atom_start[basis->bs_eptr[eid - 1] + 1];
        /* Atom not supported */
        CMS_ASSERT(basis->bs_eptr[eid - 1] != -1);
        basis->s_start_id[i] = nshells;
        for (uint32_t j = atom_start; j < atom_end; j++) {
            basis->f_start_id[nshells + j - atom_start] = nfunctions;
            basis->nexp[nshells + j - atom_start] = basis->bs_nexp[j];
            basis->xyz0[(nshells + j - atom_start) * 4 + 0] = basis->xn[i];
            basis->xyz0[(nshells + j - atom_start) * 4 + 1] = basis->yn[i];
            basis->xyz0[(nshells + j - atom_start) * 4 + 2] = basis->zn[i];
            basis->momentum[nshells + j - atom_start] = basis->bs_momentum[j];
            max_momentum = (max_momentum > basis->bs_momentum[j] ?
                max_momentum : basis->bs_momentum[j]);
            if (max_nexp < basis->bs_nexp[j]) {
                max_nexp = basis->bs_nexp[j];
                max_nexp_id = nshells + j - atom_start;
            }
            basis->cc    [nshells + j - atom_start] = basis->bs_cc[j];
            basis->exp   [nshells + j - atom_start] = basis->bs_exp[j];
            basis->minexp[nshells + j - atom_start] = vector_min(basis->nexp[nshells + j - atom_start], basis->exp[nshells + j - atom_start]);
            basis->norm  [nshells + j - atom_start] = basis->bs_norm[j];
            if (basis->basistype == SPHERICAL) {
                nfunctions += 2 * basis->bs_momentum[j] + 1;
                maxdim     = (2 * basis->bs_momentum[j] + 1) > maxdim ?
                             (2 * basis->bs_momentum[j] + 1) : maxdim;
            }
            else if (basis->basistype == CARTESIAN) {
                nfunctions += (basis->bs_momentum[j] + 1)*(basis->bs_momentum[j] + 2)/2;
                maxdim     = ((basis->bs_momentum[j] + 1)*(basis->bs_momentum[j] + 2)/2) > maxdim ?
                             ((basis->bs_momentum[j] + 1)*(basis->bs_momentum[j] + 2)/2) : maxdim;
            }
            basis->f_end_id[nshells + j - atom_start] = nfunctions - 1;
        }
        nshells += atom_end - atom_start;
    }
    basis->s_start_id[natoms] = nshells;
    basis->maxdim             = maxdim;       // max number of functions for a shell
    basis->nfunctions         = nfunctions;   // total number of functions for molecule
    basis->max_momentum       = max_momentum; // max angular momentum
    basis->max_nexp           = max_nexp;     // max number of primitives per shell
    basis->max_nexp_id        = max_nexp_id;  // index of the above
    
    return CMS_STATUS_SUCCESS;
}

/* Read the xyz file and put data into the BasisSet_t structure.
 * 
 * Net charge is read from comment line of xyz file.
 * Atom positions are converted from Angstroms to Bohr.
 * basis->eid[i] is element id, i.e., atomic number.
 * basis->charge[i] is double precision version of eid.
 * Nuclear energy is computed and stored in basis->ene_nuc;
 * Warning: check that your xyz file does not use Fortran scientific notation.
 */
CMSStatus_t CMS_import_molecule(char *file, BasisSet_t basis)
{
    FILE *fp;
    char line[1024];
    char str[1024];
    int natoms;
    int nelectrons;
    int nsc;
    int i;

    fp = fopen (file, "r");
    if (fp == NULL)
    {
        CMS_PRINTF (1, "failed to open molecule file %s\n", file);
        return CMS_STATUS_FILEIO_FAILED;
    }

    // number of atoms    
    if (fgets (line, 1024, fp) == NULL)
    {
        CMS_PRINTF (1, "file %s has a wrong format\n", file);
        return CMS_STATUS_FILEIO_FAILED; 
    }
    sscanf (line, "%d", &(basis->natoms));    
    if (basis->natoms <= 0)
    {
        CMS_PRINTF (1, "file %s has a wrong format\n", file);
        return CMS_STATUS_FILEIO_FAILED;
    }
        
    // skip comment line
    if (fgets (line, 1024, fp) == NULL)
    {
        CMS_PRINTF (1, "file %s has a wrong format\n", file);
        return CMS_STATUS_FILEIO_FAILED; 
    }
    basis->Q = atof(line);
    
    basis->xn = (double *)malloc (sizeof(double) * basis->natoms);
    basis->yn = (double *)malloc (sizeof(double) * basis->natoms);
    basis->zn = (double *)malloc (sizeof(double) * basis->natoms);
    basis->charge = (double *)malloc (sizeof(double) * basis->natoms); 
    basis->eid = (int *)malloc (sizeof(int) * basis->natoms);
    if (NULL == basis->xn ||
        NULL == basis->yn ||
        NULL == basis->zn ||
        NULL == basis->charge ||
        NULL == basis->eid)
    {
        CMS_PRINTF (1, "memory allocation failed\n");
        return CMS_STATUS_ALLOC_FAILED;
    }
    basis->mem_size += sizeof(double) * basis->natoms * 4 + sizeof(int) * basis->natoms;

    // read x, y and z
    natoms = 0;
    nelectrons = 0;
    while (fgets (line, 1024, fp) != NULL)
    {
        nsc = sscanf (line, "%s %lf %lf %lf",
                      str, &(basis->xn[natoms]), 
                      &(basis->yn[natoms]), &(basis->zn[natoms]));
        if (isalpha(str[0]))
        {
            basis->xn[natoms] = basis->xn[natoms] * A2BOHR;
            basis->yn[natoms] = basis->yn[natoms] * A2BOHR;
            basis->zn[natoms] = basis->zn[natoms] * A2BOHR;   
            if (strlen(str) > MAXATOMNAME || nsc == EOF)
            {
                CMS_PRINTF (1, "atom %s in %s is not supported\n", str, file);
                return CMS_STATUS_INVALID_VALUE;
            }
            for (i = 0; i < ELEN; i++)
            {
                if (strcmp (str, etable[i]) == 0)
                {
                    basis->eid[natoms] = i + 1;
                    break;
                }
            }
            if (i == ELEN)
            {
                CMS_PRINTF (1, "atom %s is not supported\n", str);
                return CMS_STATUS_INVALID_VALUE;
            }
            basis->charge[natoms] = (double)(basis->eid[natoms]);
            nelectrons += basis->eid[natoms];
            natoms++;
        }
    }
    basis->nelectrons = nelectrons;
    if (natoms != basis->natoms)
    {
        CMS_PRINTF (1, "file %s natoms %d does not match the header\n",
            file, natoms);
        return CMS_STATUS_FILEIO_FAILED;
    }

    // compute nuc energy
    double ene = 0.0;
    for (int A = 0; A < natoms; A++)
    {
        for (int B = A + 1; B < natoms; B++)
        {
            double dx = basis->xn[A] - basis->xn[B];
            double dy = basis->yn[A] - basis->yn[B];
            double dz = basis->zn[A] - basis->zn[B];
            double R = sqrt(dx * dx + dy * dy + dz * dz);
            ene += basis->charge[A] * basis->charge[B] / R;
        }
    }
    basis->ene_nuc = ene;
    
    fclose (fp);
    
    return CMS_STATUS_SUCCESS;
}


/* Construct the basis set data structure from the basis set file.
 * (No molecular information is used.)
 *
 * Shell information includes:
 *   bs_momentum - angular momentum 0=s, 1=p, etc.
 *   bs_cc[i][j] - contraction coefs for element i
 *   bs_exp[i][j]- orbital exponents for element i
 *
 * Handle sp shells by splitting them into separate s and p shells
 */
CMSStatus_t CMS_import_basis(char *file, BasisSet_t basis)
{
    FILE *fp;
    char line[1024];
    char str[1024];
    int natoms;
    int nshells;
    int i;
    int j;
    int nexp;
    int ns;
    double beta;
    double cc[MAXNS];
    long int mark;
    int bs_totnexp;

    fp = fopen (file, "r");
    if (fp == NULL)
    {
        CMS_PRINTF (1, "failed to open molecule file %s\n", file);
        return CMS_STATUS_FILEIO_FAILED;
    }

    // read the basis type
    if (fgets (line, 1024, fp) == NULL)
    {
        CMS_PRINTF (1, "file %s has a wrong format\n", file);
        return CMS_STATUS_FILEIO_FAILED;    
    }
    sscanf (line, "%s", str);
    if (strcmp (str, "cartesian") == 0)
    {
        CMS_PRINTF (1, "Warning: found CARTESIAN basis type in gbs file... assuming Simint\n");
        basis->basistype = CARTESIAN;
    }
    else if (strcmp (str, "spherical") == 0)
    {
        CMS_PRINTF (1, "Warning: found SPHERICAL basis type in gbs file... assuming OptErD\n");
        basis->basistype = SPHERICAL;
    }
    else
    {
        CMS_PRINTF (1, "Warning: no valid basis type in gbs file... assuming Simint\n");
        basis->basistype = CARTESIAN;
    }

    // get number of atoms (elements in the basis set)
    natoms = 0;
    nshells = 0;
    bs_totnexp = 0;
    while (fgets (line, 1024, fp) != NULL)
    {
        if (isalpha (line[0])) // found next element
        {
            natoms++;
            while (fgets (line, 1024, fp) != NULL)
            {
                if (isalpha (line[0])) // found next shell
                {
                    sscanf (line, "%s %d %lf",
                        str, &nexp, &beta);
                    ns = strlen (str); // ns=1 for S,P,...; ns=2 for SP shell
                    nshells += ns;     // total number of shells
                    bs_totnexp += ns * nexp; // total number of primitive funcs
                }
                if (line[0] == '*')
                {
                    break;
                }
            }
         }
    }
    basis->bs_natoms = natoms;       // number of elements
    basis->bs_nshells = nshells;     // number of shells in the basis set (not the molecule)
    basis->bs_totnexp = bs_totnexp;  // number of primitive functions
    basis->bs_nelements = ELEN;
    basis->bs_eptr       = (int *)malloc (sizeof(int) * basis->bs_nelements); // map atomic number to basis set entry index
    basis->bs_atom_start = (int *)malloc (sizeof(int) * (natoms + 1));        // start of element data in arrays of length nshells
    basis->bs_nexp       = (int *)malloc (sizeof(int) * nshells);             // for each shell, number of primitive functions
    basis->bs_cc         = (double **)malloc (sizeof(double *) * nshells);    // for each shell, coefs
    basis->bs_norm       = (double **)malloc (sizeof(double *) * nshells);    // for each shell, normalization constants
    basis->bs_exp        = (double **)malloc (sizeof(double *) * nshells);    // for each shell, exponents
    basis->bs_momentum   = (int *)malloc (sizeof(int) * nshells);
    basis->bs_eid        = (int *)malloc (sizeof(int) * natoms);
    CMS_ASSERT(basis->bs_eptr       != NULL);
    CMS_ASSERT(basis->bs_atom_start != NULL);
    CMS_ASSERT(basis->bs_nexp       != NULL);
    CMS_ASSERT(basis->bs_cc         != NULL);
    CMS_ASSERT(basis->bs_norm       != NULL);
    CMS_ASSERT(basis->bs_exp        != NULL);
    CMS_ASSERT(basis->bs_momentum   != NULL);
    CMS_ASSERT(basis->bs_eid        != NULL);
    basis->mem_size += sizeof(int) * (basis->bs_nelements + natoms * 2 + 1 + nshells * 2)
                     + sizeof(double*) * nshells * 3;
    for (i = 0; i < basis->bs_nelements; i++) {
        basis->bs_eptr[i] = -1;
    }

    // get nshells
    rewind (fp);
    fgets (line, 1024, fp);
    natoms = 0;
    nshells = 0;
    bs_totnexp = 0;
    while (fgets (line, 1024, fp) != NULL)
    {
        if (isalpha (line[0])) // found element
        {
            sscanf (line, "%s", str);
            for (i = 0; i < basis->bs_nelements; i++)
            {
                if (strcmp (str, etable[i]) == 0) // atomic number is i
                {
                    basis->bs_eptr[i] = natoms; // map from atomic number to basis set entry index
                    basis->bs_eid[natoms] = i;  // map from basis set entry to atomic number
                    break;
                }
            }
            if (i == basis->bs_nelements)
            {
                CMS_PRINTF (1, "atom %s in %s is not supported\n", str, file);
                return CMS_STATUS_INVALID_VALUE;
            }
            basis->bs_atom_start[natoms] = nshells; // pointer to where shells begin for this element
            natoms++;
            // read shells
            while (fgets (line, 1024, fp) != NULL)
            {
                if (isalpha (line[0]))
                {
                    sscanf (line, "%s %d %lf",
                        str, &nexp, &beta);
                    ns = strlen (str);
                    if (nexp <= 0 || ns <= 0 || ns > MAXNS)
                    {
                        CMS_PRINTF (1, "file %s contains invalid values\n", file);
                        return CMS_STATUS_INVALID_VALUE;                        
                    }
                    mark = ftell (fp);
                    for (i = 0; i < ns; i++) // usually ns==1, but ns==2 for SP shells
                    {
                        basis->bs_nexp[nshells] = nexp;
                        basis->bs_cc[nshells]   = (double *)ALIGNED_MALLOC (sizeof(double) * nexp);
                        basis->bs_exp[nshells]  = (double *)ALIGNED_MALLOC (sizeof(double) * nexp);
                        basis->bs_norm[nshells] = (double *)ALIGNED_MALLOC (sizeof(double) * nexp);
                        assert(basis->bs_cc[nshells]   != NULL);
                        assert(basis->bs_exp[nshells]  != NULL);
                        assert(basis->bs_norm[nshells] != NULL);
                        basis->mem_size += sizeof(double) * nexp * 3;
                        for (j = 0; j < SLEN; j++) // match angular momentum symbol to index
                        {
                            if (str[i] == mtable[j])
                            {
                                basis->bs_momentum[nshells] = j;
                                break;
                            }
                        }
                        if (j == SLEN)
                        {
                            CMS_PRINTF (1, "shell %s in file %s is not supported\n",
                                str, file);
                            return CMS_STATUS_INVALID_VALUE;  
                        }
                        fseek (fp, mark, SEEK_SET);
                        for (j = 0; j < basis->bs_nexp[nshells]; j++)
                        {
                            if (fgets (line, 1024, fp) == NULL ||
                                line[0] == '*' ||
                                isalpha (line[0]))
                            {
                                CMS_PRINTF (1, "file %s has a wrong format\n", file);
                                return CMS_STATUS_FILEIO_FAILED;
                            }
                            // read contraction coefs into temporary array cc
                            // and then store the element cc[i] corresponding to this shell
                            sscanf (line, "%lf %lf %lf %lf",
                                    &(basis->bs_exp[nshells][j]),
                                    &(cc[0]), &(cc[1]), &(cc[2]));
                            basis->bs_cc[nshells][j] = cc[i];
                        }
                        bs_totnexp += basis->bs_nexp[nshells];
                        nshells++;
                    }
                }
                if (line[0] == '*')
                {
                    break;
                }
            }
         }
    }
    basis->bs_atom_start[natoms] = nshells;
    if (nshells != basis->bs_nshells || basis->bs_totnexp != bs_totnexp)
    {
        CMS_PRINTF (1, "file %s has a wrong format\n", file);
        return CMS_STATUS_FILEIO_FAILED;    
    }

    fclose (fp);

    normalization (basis);
    
    return CMS_STATUS_SUCCESS;
}


/* Import SAD initial guesses.
 * (No molecular information is used.)
 *
 * basis->guess[i][*] will contain the guess for element i in the basis set.
 * Input "file" is filename for the basis set file; the guess files are assumed
 * to be in same irectory as the basis set file.  Naming convention is C.dat
 * for carbon, etc.
 *
 * If a guess file for an element is not found, or an inappropriate file is
 * found (e.g., does not match number of functions because it is for a 
 * different basis set), then a zero initial guess for that atom is used.
 */
CMSStatus_t CMS_import_guess(char *file, BasisSet_t basis)
{
    char *dir;
    if (file != NULL) {
        dir = strdup(dirname(file));
    } else {
        dir = strdup("/");
    }

    char fname[1024];
    char line[1024];    
    // pointers to guess for each element in basis set
    basis->guess = (double **)malloc (sizeof(double *) * basis->bs_natoms); 
    if (basis->guess == NULL)
    {
        return CMS_STATUS_ALLOC_FAILED;
    }
     
    // loop over elements in the basis set
    for (int i = 0; i < basis->bs_natoms; i++)
    {
        const int atom_start = basis->bs_atom_start[i];
        const int atom_end = basis->bs_atom_start[i + 1];
        int eid = basis->bs_eid[i];
        // calculate number of functions for this atom
        int nfunctions = 0;
        for (int j = atom_start; j < atom_end; j++)
        {
            if (basis->basistype == SPHERICAL)
            {
                nfunctions +=
                    2 * basis->bs_momentum[j] + 1;
            }
            else if (basis->basistype == CARTESIAN)
            {
                nfunctions +=
                    (basis->bs_momentum[j] + 1)*(basis->bs_momentum[j] + 2)/2;
            }
        }
        // allocate space for guess for element i (not a global-sized matrix)
        basis->guess[i] =
            (double *)malloc(sizeof(double) * nfunctions * nfunctions);
        CMS_ASSERT(basis->guess[i] != NULL);
        basis->mem_size += sizeof(double) * nfunctions * nfunctions;

        // read guess
        eid = (eid >= ELEN ? 0 : eid);
        sprintf(fname, "%s/%s.dat", dir, etable[eid]);
        FILE *fp = fopen(fname, "r");
        int flag = 0;
        if (fp != NULL) {
            CMS_PRINTF (1, "Found SAD file for %s\n", etable[eid]);
            for (int j = 0; j < nfunctions * nfunctions; j++) {
                if (fgets (line, 1024, fp) == NULL) {
                    CMS_PRINTF (1, "Bad SAD file\n");
                    flag = 1;
                    goto end;
                }
                sscanf (line, "%le", &(basis->guess[i][j]));
            }
            // test symmetry
            for (int j = 0; j < nfunctions; j++) {
                for (int k = 0; k < nfunctions; k++) {
                    if (basis->guess[i][j * nfunctions + k] -
                        basis->guess[i][k * nfunctions + j] > 1e-12) {
                        CMS_PRINTF (1, "Bad SAD matrix: not symmetric\n");
                        flag = 1;
                        goto end;
                    }
                }
            }
        } else {
            flag = 1;
        }
end:        
        if (flag == 1) {
            memset(basis->guess[i], 0,
                sizeof(double) * nfunctions * nfunctions);          
        }
        else
        {
            CMS_PRINTF (1, "Using SAD file for %s\n", etable[eid]);
        }
    }
    
    if (file != NULL) {
        free(dir);
    }
    return CMS_STATUS_SUCCESS;
}

CMSStatus_t CMS_loadChemicalSystem(BasisSet_t basis, char *bsfile, char *molfile)
{
    CMSStatus_t status;

    // read xyz file
    status = CMS_import_molecule(molfile, basis);
    if (status != CMS_STATUS_SUCCESS) return status;

    // read basis set
    status = CMS_import_basis(bsfile, basis);
    if (status != CMS_STATUS_SUCCESS) return status;
    
    // parse xyz
    status = CMS_parse_molecule(basis);
    if (status != CMS_STATUS_SUCCESS) return status;

    // import guess
    status = CMS_import_guess(bsfile, basis);
    if (status != CMS_STATUS_SUCCESS) return status;

    printf("CMS basis set memory usage = %.2lf MB\n", basis->mem_size / 1048576.0);
    
    return CMS_STATUS_SUCCESS;
}

int CMS_getNumAtoms(BasisSet_t basis)
{
    return basis->natoms;
}

int CMS_getNumShells(BasisSet_t basis)
{
    return basis->nshells;
}

int CMS_getNumFuncs(BasisSet_t basis)
{
    return basis->nfunctions;
}

int CMS_getNumOccOrb(BasisSet_t basis)
{
    return ((basis->nelectrons - basis->Q)/2);
}

int CMS_getFuncStartInd(BasisSet_t basis, int shellid)
{
    return basis->f_start_id[shellid];
}

int CMS_getFuncEndInd(BasisSet_t basis, int shellid)
{
    return basis->f_end_id[shellid];
}

int CMS_getShellDim(BasisSet_t basis, int shellid)
{
    return (basis->f_end_id[shellid] - basis->f_start_id[shellid] + 1);
}

int CMS_getMaxShellDim(BasisSet_t basis)
{
    return basis->maxdim;
}

int CMS_getAtomStartInd(BasisSet_t basis, int atomid)
{
    return basis->s_start_id[atomid];
}

int CMS_getAtomEndInd(BasisSet_t basis, int atomid)
{
    return basis->s_start_id[atomid + 1] - 1;
}

int CMS_getTotalCharge(BasisSet_t basis)
{
    return basis->Q;
}

int CMS_getNneutral(BasisSet_t basis)
{
    return basis->nelectrons;
}

int CMS_getMaxMomentum(BasisSet_t basis)
{
    return basis->max_momentum;
}

int CMS_getMaxPrimid(BasisSet_t basis)
{
    return basis->max_nexp_id;
}

int CMS_getMaxnumExp(BasisSet_t basis)
{
    return basis->max_nexp;   
}

double CMS_getNucEnergy(BasisSet_t basis)
{
    return basis->ene_nuc;
}

void CMS_getInitialGuess(BasisSet_t basis, int atomid, double **guess, int *spos, int *epos)
{
    const int eid = basis->eid[atomid] - 1;
    const int start_shell = basis->s_start_id[atomid];
    const int end_shell = basis->s_start_id[atomid + 1];
    *guess = basis->guess[basis->bs_eptr[eid]];
    *spos  = basis->f_start_id[start_shell];
    *epos  = basis->f_end_id[end_shell - 1];
}

void CMS_getShellxyz(BasisSet_t basis, int shellid, double *x, double *y, double *z)
{
    *x = basis->xyz0[shellid*4 + 0];
    *y = basis->xyz0[shellid*4 + 1];
    *z = basis->xyz0[shellid*4 + 2];
}
