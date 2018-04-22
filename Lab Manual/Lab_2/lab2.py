#Write a program to generate a dictionary that contains (i,sqrt(i)), where i is an integer between 1 and n. n is a number input by the user.

#import math;
#
#n = int(input());
#dic = {};
#
#for i in range(1,n+1):
#    dic[i] = float(format(math.sqrt(i), '.2f'));
#print(dic);

######################################################################

#Write a simple calculator program using functions add, sub, mul and div. The program should accepts two numbers and an operator and calls the corresponding function to perform the operation.

#num1 = int(input('type first number\n'));
#num2 = int(input('type second number\n'));
#opr = input('type operator\n');
#
#def add():
#    return num1 + num2;
#
#def sub():
#    return num1 - num2;
#
#def mul():
#    return num1 * num2;
#
#def div():
#    return num1 / num2;
#
#def default():
#    return 'invalid input';
#
#switch = {
#        '+': add(),
#        '-': sub(),
#        '*': mul(),
#        '/': div()
#        }
#
#print(switch.get(opr, default()));

###############################################################################

#Write a function that generates a list with values that are square of number between 1 and 20

#squares = [];
#
#for num in range(1,21):
#    squares.append(num**2)
#print(squares)

###############################################################################

#Define a class named Shape with static method printType. 
#Define methods draw() and area(). Now define two class Rectangle and Triangle.
#Rectangle has two attributes length and width. The Triangle class has attributes a,b and c.  Override the two methods of shape class. 
#Demonstrate the functionality of class by creating its objects.

#class Shape():
#    
#    @staticmethod
#    def printType():
#        print('static method called');
#    
#    def draw(self):
#        print('draw method called of Shape class');
#    
#    def area(self):
#        print('area method called of Shape class');
#    
#class Rectangle(Shape):
#    
#    width = 0;
#    length = 0;
#    
#    def draw(self):
#        print('draw method called of Rectangle class');
#    
#    def area(self):
#        print('area method called of Rectangle class');
#
#class Triangle(Shape):
#    
#    a = 0;
#    b = 0;
#    c = 0;
#    
#    def draw(self):
#        print('draw method called of Triangle class');
#    
#    def area(self):
#        print('area method called of Triangle class');
#
#obj = Rectangle();
#obj.draw()
#obj = Triangle();
#obj.printType()
#obj = Shape();
#obj.draw()

##############################################################################

# Using recursion, write a program to calculate the reverse of a string.

#sentance = list(input());
#reverseString = [];
#counter = 10;
#def reversing():
#    global counter;
#    reverseString.append(sentance[counter]);  
#    if counter == 0:
#        return print(''.join(reverseString));
#    counter -= 1;
#    return reversing();
#
#reversing();



























































