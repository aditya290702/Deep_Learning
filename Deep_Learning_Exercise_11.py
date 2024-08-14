import numpy as np
def conv_operation():

        #Taking an example image,Specifying the dimensions and shape
        image = np.array([[3,0,1,2,7,4],
                         [1,5,8,9,3,1],
                         [2,7,2,5,1,3],
                         [0,1,3,1,7,8],
                         [4,2,1,6,2,8],
                         [2,4,5,2,3,9]])
        n = image.shape
        n_r = n[0]
        n_c = n[1]
        print("Image rows :",n_r)
        print("Image cols :",n_c)

        #Taking an example Filter,Specifying the dimensions and shape
        filter = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        f = filter.shape
        f_r = f[0]
        f_c = f[1]
        print("Filter rows :",f_r)
        print("Filter cols :",f_c)
        print()
        print("-----------------------------------")
        print()

        #Creating and empty values(Zero valued)np array bu the formula of N-F+1
        result_rows = (n_r-f_r) + 1
        result_cols = (n_c - f_c) + 1
        print("Result image rows after filter strides :", result_rows)
        print("Result image cols after filter strides:",result_cols)
        zero_array = np.zeros((result_rows,result_cols))
        print()



        print("Image :")
        print()
        print(image)
        print()
        print("-----------------------------------")
        print()
        print("Image dimensions:",image.shape)
        print()
        print("Filter :")
        print(filter)
        print()
        print("-----------------------------------")
        print()
        print("Filter dimensions:", filter.shape)
        print()

        stride_arr = []

        #Setting a nested loop to create blocks(STRIDES) according
        #to the dimensions of the filter
        print("STRIDES:")
        print()
        for i in range(0,f_r+1):
                for j in range(0,f_c+1):
                        stride = image[i:i+3,j:j+3]
                        stride_arr.append(stride)
                        print(stride)
                        print()
        print("-----------------------------------")
        print()

        result_array = []

        for k in range(0,len(stride_arr)):
                result = np.sum((stride_arr[k] * filter))
                result_array.append(result)



        result_image = np.array(result_array).reshape((4,4))
        print("Result Image :")
        print(result_image)
        print()

        print("---------------------------------------")


        return image, result_image, result_array, result_cols, result_rows, filter,f_c,f_r,n_c,n_r
def Maxpool(image, result_array, result_cols, result_rows, filter,f_c,f_r,n_c,n_r):
        conv_operation()

        result_image = np.array([[1,3,2,1,3],
                          [2,9,1,1,5],
                          [1,3,2,3,2],
                          [8,3,5,1,0],
                          [5,6,1,2,9]])


        n_1 = result_image.shape
        n_r_1 = n_1[0]
        n_c_1 = n_1[1]


        result_rows_1 = (n_r_1 - f_r) + 1
        result_cols_1 = (n_c_1 - f_c) + 1



        print()
        print("Strides for Max-pool done on the output image")
        print()
        print("Image :")
        print()
        print(result_image)
        print()
        print("----------------------------------")

        print("Strides")
        print("----------------------------------")

        Stride1_arr = []
        Maxpool_vals = []
        for i in range(0,result_rows_1):
                for j in range(0,result_cols_1):
                        Stride1 = result_image[i:i+f_r,j:j+f_c]
                        Stride1_arr.append(Stride1)
                        print(Stride1)
                        print()

        for j in range(0,len(Stride1_arr)):
                Zero_matrix = (np.max(Stride1_arr[j]))
                Maxpool_vals.append(Zero_matrix)
        # print(Maxpool_vals)
        print("---------------------------------------")
        print()

        Maxpool_image = np.array(Maxpool_vals).reshape((3, 3))
        print("Maxpooling : ")
        print(Maxpool_image)









def main():
        image, result_image, result_array, result_cols, result_rows, filter,f_c,f_r,n_c,n_r = conv_operation()
        Maxpool(image, result_array, result_cols, result_rows, filter,f_c,f_r,n_c,n_r)

main()