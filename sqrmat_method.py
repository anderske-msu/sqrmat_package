import numpy as np

# TODO install package into enviroment
import PyTPSA   # * Must be in the same folder as PyTPSA or in the path

# TODO install these as packages
# ! Import sys and run ' sys.path.insert(1, '../Square-Matrix/Custom_Libraries/') ' and ' sys.path.insert(1, '../Library/CodeLib/') '
from sqrmat import jordan_chain_structure, tpsvar
from commonfuncs import numpify

class square_matrix:

    # TODO Add tracking
    
    def __init__(self, dim, order):
        self.dim = dim
        self.order = order
        
        self._hp = tpsvar(dim, order=order)
        # self.variables = self._hp.get_variables()
        self.variables = self._hp.vars
        
        self.degenerate_list = [None for i in range(dim)]
        self.left_vector = [None for i in range(dim)]

        self.jordan_norm_form_matrix = [None for i in range(dim)]
        self.jordan_chain_structure = [None for i in range(dim)]
        
        self.square_matrix = None

        self.__fztow = [None for i in range(dim)]
        self.__wftoz = [None for i in range(dim)]
        self.__fzmap = [None for i in range(dim)]

    def construct_square_matrix(self, periodic_map):
        self.__fzmap = [numpify(f) for f in periodic_map]
        self._hp.construct_sqr_matrix(periodic_map=periodic_map)
        self.square_matrix = self._hp.sqr_mat

    def get_transformation(self, target=None, res=None, chain=0, dim=None, epsilon=1e-15):

        # Check if contruct sqrmat was run
        if self.square_matrix is None:
            raise Exception("The square matrix has not been constructed. Run \"construct_square_matrix\" first.")

        # First get the degenerate list
        if target is None:
            target = [i for i in range(1, self.dim+1)]
        
        # TODO Does this do what I want it to do?
        if dim is None:
            dim_iter = self.dim
        else:
            dim_iter = dim

        for di in range(dim_iter):
            self.degenerate_list[di] = self._hp.get_degenerate_list(di+1, resonance=res)
            self._hp.sqrmat_reduction()
            self._hp.jordan_form()

            self.jordan_norm_form_matrix[di] = self._hp.jnf_mat
            self.left_vector[di] = self._hp.left_vector[chain]
            self.jordan_chain_structure[di] = jordan_chain_structure(self.jordan_norm_form_matrix[di], epsilon=epsilon)

        wx0z = PyTPSA.tpsa(input_map=self.left_vector[0], dtype=complex)
        wx0cz = wx0z.conjugate(mode='CP')

        wy0z = PyTPSA.tpsa(input_map=self.left_vector[1], dtype=complex)
        wy0cz = wy0z.conjugate(mode='CP')

        w0list = [wx0z, wx0cz, wy0z, wy0cz]

        invw0z = PyTPSA.inverse_map(w0list)

        self.__fztow = [numpify(f) for f in w0list]
        self.__wftoz = [numpify(f) for f in invw0z]

        # Build Jacobian function
        self.__fjacobian = [numpify(wtemp.derivative(i+1)) for wtemp in w0list for i in range(2*self.dim)]


    def w(self, z):
        """Transforms normalized complex coordinates into the transformed phase space.
        
        Inputs:
            z: Array-like; [zx, zx*, zy, zy*]

        Returns:
            w; Numpy Array; [wx, wx*, wy, wy*]
        """
        
        if self.__fztow[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__fztow[0](z), self.__fztow[1](z), self.__fztow[2](z), self.__fztow[3](z)])

    def z(self, w):
        """Tranformes transformed coordinates into the normalized complex coordinate phase space.
        
        Inputs:
            w: Array-like; [wx, wx*, wy, wy*]

        Returns:
            z; Numpy Array; [zx, zx*, zy, zy*]
        """

        if self.__wftoz[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__wftoz[0](w), self.__wftoz[1](w), self.__wftoz[2](w), self.__wftoz[3](w)])

    def jacobian(self, z):
        """Returns the value of the jacobian matrix at z.
        
        Inputs:
            z: Array-like; [zx, zx*, zy, zy*]
            
        Returns:
            jac: Numpy Array; Jacobian
        """
        jac=[]
        
        for f in self.__fjacobian:
           jac.append(f(z))
        
        return np.array(jac)


    def map(self, z):
        """Runs z through the given one turn map. z' = f(z)
        
        Inputs:
            z; Array-like; [zx, zx*, zy, zy*]

        Returns:
            z'; Numpy Array; [zx', zx'*, zy', zy'*]
        """
        if self.__fzmap[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__fzmap[0](z), self.__fzmap[1](z), self.__fzmap[2](z), self.__fzmap[3](z)])

    def test_sqrmat(self, atol: float = 1e-8, results: bool = False) -> tuple:
        """Checks that the lower triagular elements of the square matrix are close to zero.
        
        Inputs:
            atol: The absolute tolerance to check against.
            results: Print results

        Returns:
            pass: If all the points are close to zero the funtion will return True
        """

        dimx, dimy = self.square_matrix.shape

        pass_ = False

        num_errors = 0
        max_error = 0

        for i in range(dimx):
            for j in range(dimy):
                term = np.abs(self.square_matrix[i,j])

                if i > j:                                       # On the lower diagonal
                    if not np.isclose(term, 0, atol=atol):      # If the element is not close to zero count it
                        num_errors += 1

                    if term > max_error:                        # Track the largest element
                        max_error = term*1

        if results:
            print("Number of Errors:", num_errors)
            print("Max Error:", max_error)

        if num_errors == 0:
            pass_ = True
  
        return pass_, num_errors, max_error