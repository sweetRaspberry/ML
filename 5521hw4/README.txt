Luze Yang
yang5503@umn.edu
5329111
Instructor: Arindam Banerjee

To run the code, put all python files in the same folder, type "python3 q3.py" in terminal.

The input data will be normalized in my_cross_val function

There is a parameter "diag" for MultiGaussClassify to determine whether the covariance is full or diagonal. I set the default "diag=False", which means full covariance

I have changed my_cross_val(method,X,y,k) function be my_cross_val(method,X,y,num_folders), to avoid confusion about "k" (which means number of class) in MultiGaussClassify
