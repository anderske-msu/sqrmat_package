import numpy as np
import sympy as sp      # For test_u

# TODO install package into enviroment
import PyTPSA   # * Must be in the same folder as PyTPSA or in the path

# TODO install these as packages
# ! Import sys and run ' sys.path.insert(1, '../Square-Matrix/Custom_Libraries/') ' and ' sys.path.insert(1, '../Library/CodeLib/') '
from sqrmat import jordan_chain_structure, tpsvar
from commonfuncs import numpify

class square_matrix:

    def __init__(self, dim: int, order: int):
        self.dim = dim                                              # Number of spacial dimentions
        self.order = order                                          # Square Matrix order
        
        self._hp = tpsvar(dim, order=order)
        self.variables = self._hp.vars
        
        self.degenerate_list = [None for _ in range(dim)]
        self.left_vector = [None for _ in range(dim)]
        self.left_vector_second = [None for _ in range(dim)]

        self.jordan_norm_form_matrix = [None for _ in range(dim)]
        self.jordan_chain_structure = [None for _ in range(dim)]

        self.__weights = [[1,0,0,0],[0,0,1,0]]
        
        self.square_matrix = None

        self.__fztow = [None for _ in range(dim)]
        self.__wftoz = [None for _ in range(dim)]
        self.__fzmap = [None for _ in range(dim)]

    def construct_square_matrix(self, periodic_map: list):
        self.__fzmap = [numpify(f) for f in periodic_map]
        self._hp.construct_sqr_matrix(periodic_map=periodic_map)
        self.square_matrix = self._hp.sqr_mat

    def get_transformation(self, res=None, left_eigenvector: int = 0, epsilon: float = 1e-15):
        """
        Creates the transformation for the approximate action.

        Inputs:
            res: list
                The resonance the linear tunes are on. Default is None
            
            left_eigenvector: int
                The left eigenvector (row of U) in the first jordan chain to be used. Default is the first (0).

            epsilon: float
                Epsilon for the finding the jordan chain structure. Default is 1e-15.

        Outputs:
            None
        """

        # TODO Allow the use of mulitple Jordan Chains in the transformation

        # Check if contruct sqrmat was run
        if self.square_matrix is None:
            raise Exception("The square matrix has not been constructed. Run \"construct_square_matrix\" first.")

        for di in range(self.dim):
            self._hp.get_degenerate_list(di+1, resonance=res)
            self._hp.sqrmat_reduction()
            self._hp.jordan_form()

            # ? Are these Better as lists or np.ndarrays?
            self.degenerate_list[di] = np.array(self._hp.degenerate_list.tolist())
            self.jordan_norm_form_matrix[di] = np.array(self._hp.jnf_mat.tolist())
            self.left_vector[di] = self._hp.left_vector[left_eigenvector].tolist()
            self.left_vector_second[di] = self._hp.left_vector[left_eigenvector+1].tolist() # TODO make left_eig a list to choose (not plus 1)
            self.jordan_chain_structure[di] = jordan_chain_structure(self.jordan_norm_form_matrix[di], epsilon=epsilon)

        # TODO Rewrite to make sure it'll work with dim != 2

        # First left eigenvector
        wx0z = PyTPSA.tpsa(input_map=self.left_vector[0], dtype=complex)
        wx0cz = wx0z.conjugate(mode='CP')

        wy0z = PyTPSA.tpsa(input_map=self.left_vector[1], dtype=complex)
        wy0cz = wy0z.conjugate(mode='CP')

        # Second left eigenvector
        wx1z = PyTPSA.tpsa(input_map=self.left_vector_second[0], dtype=complex)
        wx1cz = wx0z.conjugate(mode='CP')

        wy1z = PyTPSA.tpsa(input_map=self.left_vector_second[1], dtype=complex)
        wy1cz = wy0z.conjugate(mode='CP')

        self.__w0list = [wx0z, wx0cz, wy0z, wy0cz]
        self.__w1list = [wx1z, wx1cz, wy1z, wy1cz]

        invw0z = PyTPSA.inverse_map(self.__w0list)

        self.__fztow = [numpify(f) for f in self.__w0list]
        self.__wftoz = [numpify(f) for f in invw0z]

        # Build Jacobian function
        # ! Used for netwons inverse, needs to be update for using weights
        self.__fjacobian = [numpify(wtemp.derivative(i+1)) for wtemp in self.__w0list for i in range(2*self.dim)]


    def w(self, z: list, weights: list = None) -> np.ndarray:
        """Transforms normalized complex coordinates into the transformed phase space.
        
        Inputs:
            z: Array-like
                [zx, zx*, zy, zy*]

            weights: list
                The weights for each left vector transformation to be used in the full w transformation. Dimentions are (dim, 4).
                The default is wj = wj0 where j is each of the spatial dimentions.
                    i.e. wj(z) = weights[j,0] * wx0(x) + weights[j,1] * wx1(x) + weights[j,2] * wy0(x) + weights[j,3] * wy1(x)

        Returns:
            w: Numpy Array
                [wx, wx*, wy, wy*] 
        """
        
        if self.__fztow[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")
        
        if (weights is not None) and (self.__weights != weights):
            
            print("Transformation weights are being updated.")
            
            # Checks if the weights have changed
            self.__weights = np.array(weights).tolist()     # TODO figure out a better way to make sure it's not passed by reference

            # Finding new transformation and inverse functions

            wz = []     # PyTPSA w terms

            for di in range(self.dim):
                wz.append(self.__weights[di,0]*self.__w0list[0] + self.__weights[di,1]*self.__w1list[0] 
                + self.__weights[di,2]*self.__w0list[2] + self.__weights[di,3]*self.__w1list[2])
            
            wcz = [wi.conjugate(mode='CP') for wi in wz]        # Conjugate terms
            
            wlist = []                                          # New w list

            for wi, wci in zip(wz, wcz):
                # Adds new W PyTPSA terms alternating between w and w*
                wlist.append(wi)
                wlist.append(wci)

            invz = PyTPSA.inverse_map(wlist)        # New inverse function

            # Update ztow functions and inverse
            self.__fztow = [numpify(f) for f in wlist]
            self.__wftoz = [numpify(f) for f in invz]
    

        return np.array([self.__fztow[0](z), self.__fztow[1](z), self.__fztow[2](z), self.__fztow[3](z)])

    def z(self, w: list) -> np.ndarray:
        """Tranformes transformed coordinates into the normalized complex coordinate phase space.
        Assumes the weights are the same ones used in the last call of self.w
        
        Inputs:
            w: Array-like
                [wx, wx*, wy, wy*]

        Returns:
            z: Numpy Array
                [zx, zx*, zy, zy*]
        """

        if self.__wftoz[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")

        return np.array([self.__wftoz[0](w), self.__wftoz[1](w), self.__wftoz[2](w), self.__wftoz[3](w)])

    def jacobian(self, z: np.ndarray) -> np.ndarray:
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


    def map(self, z: np.ndarray) -> np.ndarray:
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

    def test_u(self):
        """Tests U U^(-1) = I"""
        
        if self.__fzmap[0] is None:
            raise Exception("The transformation has not been found. Run \"get_transformation\" first.")
        
        zx, zy = sp.symbols(r'z_x, z_y')
        zxc = zx.conjugate()
        zyc = zy.conjugate()
        
        z = (zx, zxc, zy, zyc)
        
        w = self.w([*z]).tolist()
        
        ztest = self.z(w).tolist()

        for i in range(self.dim):
            ztest[i] = sp.Poly(ztest[i], z).as_dict()

        return ztest
