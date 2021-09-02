# Homework-2 Programming Exercises - Python

Acknowledgement and reference: [Python challenge programming exercises.](https://github.com/zhiwehu/Python-programming-exercises).  
20 Questions (5 score for each) are selected.

### Question 1

Question:
Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5, between 2000 and 3200 (both included).
The numbers obtained should be printed in a comma-separated sequence on a single line.

Hints: 
Consider use range(#begin, #end) method


### Question 2

Question:
Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single line.
Suppose the following input is supplied to the program:  
8  
Then, the output should be:  
40320

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

### Question 3

Question:
With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number between 1 and n (both included). and then the program should print the dictionary.
Suppose the following input is supplied to the program:
8
Then, the output should be:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
Consider use dict()

### Question 4

Question:
Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
Suppose the following input is supplied to the program:  
34,67,55,33,12,98  
Then, the output should be:
['34', '67', '55', '33', '12', '98']  <br />
('34', '67', '55', '33', '12', '98')

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
tuple() method can convert list to tuple

### Question 5

Question:
Define a class which has at least two methods:  
getString: to get a string from console input  
printString: to print the string in upper case.  
Also please include simple test function to test the class methods.

Hints:
Use __init__ method to construct some parameters

### Question 6

Question:
Write a program that calculates and prints the value according to the given formula:  
Q = Square root of [(2 * C * D)/H]  
Following are the fixed values of C and H:  
C is 50. H is 30.  
D is the variable whose values should be input to your program in a comma-separated sequence.  
Example:  
Let us assume the following comma separated input sequence is given to the program:   
100,150,180  
The output of the program should be: 
18,22,24

Hints:
If the output received is in decimal form, it should be rounded off to its nearest value (for example, if the output received is 26.0, it should be printed as 26)  
In case of input data being supplied to the question, it should be assumed to be a console input. 

### Question 7

Question:
Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i*j.  
Note: i=0,1.., X-1; j=0,1,.., Y-1.  
Example:  
Suppose the following inputs are given to the program:  
3,5  
Then, the output of the program should be:  
[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]] 

Hints:
Note: In case of input data being supplied to the question, it should be assumed to be a console input in a comma-separated form.

### Question 8

Question:
Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.    
Suppose the following input is supplied to the program:    
without,hello,bag,world    
Then, the output should be:    
bag,hello,without,world

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

### Question 9

Question:
Write a program that accepts sequence of lines as input and prints the lines after making all characters in the sentence capitalized.
Suppose the following input is supplied to the program:  
Hello world  
Practice makes perfect  
Then, the output should be:
HELLO WORLD  
PRACTICE MAKES PERFECT

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

### Question 10

Question:
Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words and sorting them alphanumerically.  
Suppose the following input is supplied to the program:  
hello world and practice makes perfect and hello world again  
Then, the output should be:  
again and hello makes perfect practice world

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.  
We use set container to remove duplicated data automatically and then use sorted() to sort the data.

### Question 25

Question:
Define a class, which have a class parameter and have a same instance parameter.

Hints:
Define a instance parameter, need add it in __init__ method  
You can init a object with construct parameter or set the value later

### Question 35
Define a function which can generate a dictionary where the keys are numbers between 1 and 20 (both included) and the values are square of keys. The function should just print the values only.

Hints:
Use dict[key]=value pattern to put entry into a dictionary.  
Use ** operator to get power of a number.  
Use range() for loops.  
Use keys() to iterate keys in the dictionary. Also we can use item() to get key/value pairs.

### Question 36
Define a function which can generate a dictionary where the keys are numbers between 1 and 20 (both included) and the values are square of keys. The function should just print the keys only.

Hints:
Use dict[key]=value pattern to put entry into a dictionary.  
Use ** operator to get power of a number.  
Use range() for loops.  
Use keys() to iterate keys in the dictionary. Also we can use item() to get key/value pairs.

### Question 37
Define a function which can generate and print a list where the values are square of numbers between 1 and 20 (both included).

Hints:
Use ** operator to get power of a number.   
Use range() for loops.  
Use list.append() to add values into a list.

### Question 43
Write a program to generate and print another tuple whose values are even numbers in the given tuple (1,2,3,4,5,6,7,8,9,10). 

Hints:
Use "for" to iterate the tuple  
Use tuple() to generate a tuple from a list.

### Question 51
Define a class named American and its subclass NewYorker. 

Hints:
Use class Subclass(ParentClass) to define a subclass.

### Question 53
Define a class named Rectangle which can be constructed by a length and width. The Rectangle class has a method which can compute the area. 

Hints:
Use def methodName(self) to define a method.

### Question 54
Define a class named Shape and its subclass Square. The Square class has an init function which takes a length as argument. Both classes have a area function which can print the area of the shape where Shape's area is 0 by default.

Hints:
To override a method in super class, we can define a method with the same name in the super class.

### Question 56
Write a function to compute 5/0 and use try/except to catch the exceptions.

Hints:
Use try/except to catch exceptions.

### Question 94
With a given list [12,24,35,24,88,120,155,88,120,155], write a program to print this list after removing all duplicate values with original order reserved.

Hints:
Use set() to store a number of values without duplicate.
