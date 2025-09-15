# Metadata:
# Author: Syed Raza
# email: sar0033@uah.edu

# import statements
import torch # type: ignore
tol = 1e-7

# This is the program to perform matrix operations on pytorch based arrays/matrices

def float_equality(n1, n2):
    """
    Params:
    (1) n1 -- type torch.float32
        the first number for equality check
    (2) n2 -- type torch.float32
        the second number for equality check
    (3) tol -- the tolerance for float equality

    Returns:
        true if they are equal
        False if they are not 
    """
    return abs(n1 - n2) < tol


def rowswap(M, src, trg):
    """
    Params: (1) A Pytorch based matrix -- M, 
        (2) index of type int the source row -- src, 
        (3) Index of type int target row -- trg

    Returns: Returns the matrix after swapping the content of source row with the target row
    """
    # swap the rows you wanted:
    M[[src, trg]] = M[[trg, src]]

    return M

def rowscale(M, src, sca):
    """
    Params:
    (1) A Pytorch based matrix -- M
    (2) index (type int) of the row to be scaled -- src
    (2) scaling factor of type int -- sca
    """
    M[src] *= sca

    return M

def rowreplacement(M, r1, r2, sca1, sca2):
    """
    Params: 
    (1) M -- type Pytorch tensor
        This is the matrix where we are applying the row replacement
    (2) r1 -- type int 
        The index of the first row
    (3) r2 -- type int 
        The index of the second row 
    (4) sca1 -- type int
        the scaling factor row 1
    (5) sca2 -- type int 
        the scalikng factor row 2
    
    Returns:
    The matrix but with finished row replacement
    """
    # clone the matrix, so we do not change the matrix in place
    M1 = M.clone()

    # scale both rows in the clone
    M1 = rowscale(M1, r1, sca1)
    M1 = rowscale(M1, r2, sca2)

    # perform row-replacement
    M[r1] = M1[r1] + M1[r2]

    # return M
    return M

# A routine for converting matrices into reduced row echelon form:
def rref(M):
    """
    Params: M -- type Pytorch tensor
        This is the matrix where we are trying to convert into Row Echelon form
    Returns: M -- type Pytorch tensor
        The row echelon version of the matix the was inputed. It used the functions generated 
        in this module: float inequality, rowswap, rowreplacement, and rowscale
    """
    # get the rows and the columns
    nrow = M.shape[0]
    ncol = M.shape[1]
    
    # The workflow is given as follows:

    row = 0
    for col in range(ncol):
        if row >= nrow:
            break
        # (a) if there is a zero in the row, rowswap with a new row
        if float_equality(M[row, col], 0): 
            # go through rest of the rows
            for k in range(row, nrow):
                if not float_equality(M[k, col], 0):
                    rowswap(M, k, row) # this is the row that includes the pivot
                    break # break rowswapping first time this happens
        
        # (b) use row scale to make the pivot element 1 
        if not float_equality(M[row, col], 0):
            rowscale(M, row, 1 / M[row, col])

        # (c) make everything below zero using row_replacement function
        for r in range(row + 1, nrow):
            if not float_equality(M[r, col], 0):
                mul = - M[r, col] / M[row, col]
                rowreplacement(M, r, row, 1.0, mul)

        row += 1

    # scale the last row separately:
    last = nrow - 1
    for col in range(ncol):
        if not float_equality(M[last, col], 0.0):
            rowscale(M, last, 1 / M[last, col])

    # make really smal numbers resulting due to floating point operations zero
    # before returning the matrix
    M[torch.abs(M) < 1e-5] = 0.0

    return M

# main function for testing the code and analyzing the reuslts:
if __name__ == "__main__":
    M = torch.tensor([[0, 3, -6, 6, 4], [3, -7, 8, -5, 8], [3, -9, 12, -9, 6]], dtype = torch.float32)

    # testing phase of the rref function:
    M = rref(M)
    print(M)