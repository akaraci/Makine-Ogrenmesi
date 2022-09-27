# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:08:03 2022

@author: karaci
"""
class Employee:
    empCount = 0
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1
   
    def displayCount(self):
       print ("Total Employee %d" % Employee.empCount) 

    def displayEmployee(self):
       print ("Name : ", self.name,  ", Salary: ", self.salary)

emp1 = Employee("Umit", 1500)
emp1.displayCount()
emp1.displayEmployee()

emp2 = Employee("Murat", 2300)
emp2.displayCount()
emp2.displayEmployee()

myObjArray=[]
myObjArray.append(emp1)
myObjArray.append(emp2)
print("\n\n")
for emp in myObjArray:
    print("Name: %s Salary:%d"%(emp.name,emp.salary))