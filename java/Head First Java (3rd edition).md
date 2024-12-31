*   [Breaking the Surface](#breaking-the-surface)
*   [A Trip to Objectville](#a-trip-to-objectville)
*   [Know Your Variables](#know-your-variables)
*   [How Objects Behave](#how-objects-behave)
*   [Extra Strength Methods](#extra-strength-methods)
*   [Using the Java Library](#using-the-java-library)
*   [Better Living in Objectville](#better-living-in-objectville)
*   [Serious Polymorphism](#serious-polymorphism)
*   [Life and Death of an Object](#life-and-death-of-an-object)
*   [Numbers Matter](#numbers-matter)
*   [Data Structures](#data-structures)
*   [Lambdas and Streams](#lambdas-and-streams)
*   [Risky Behavior](#risky-behavior)
*   [A Very Graphic Story](#a-very-graphic-story)
*   [Work on Your Swing](#work-on-your-swing)
*   [Saving Objects (and Text)](#saving-objects-and-text)
*   [Make a Connection](#make-a-connection)
*   [Dealing with Concurrency Issues](#dealing-with-concurrency-issues)

# Breaking the Surface

#### The Way Java Works
- Java is compiled into **bytecode** by the compiler, making it platform-independent.
- The bytecode runs on the **Java Virtual Machine (JVM)**, which translates it into machine code specific to the operating system.
- Compilation steps:
  1. Write the source code in `.java` files.
  2. Use the `javac` compiler to compile code into `.class` files containing bytecode.
  3. Run the program using the `java` command, which invokes the JVM.

#### Code Structure in Java
- Basic structure includes:
  - **Class**: Defines the blueprint of an object.
  - **Method**: Defines the actions/behavior of objects.
  - `main()` Method:
    - Entry point for all Java applications.
    - Syntax: 
      ```java
      public static void main(String[] args) {
          // Code here
      }
      ```
  - **Semicolons (`;`)**: End statements.

#### Why Java?
- Features that make Java popular:
  - **Object-oriented**: Focuses on objects and their interactions.
  - **Platform-independent**: Write once, run anywhere.
  - **Automatic memory management**: Garbage collection eliminates memory leaks.
  - **Secure and robust**: Type-checking and runtime error detection.

#### Looping and Branching
- **Loops**: Execute code repeatedly.
  - `for`, `while`, `do-while`.
- **Conditional Branching**: Makes decisions.
  - `if`, `else`, `switch`.

#### Example
```java
String[] wordList = {"innovative", "scalable", "next-generation"};
// The Math.random method returns a number from zero to just less than one.
String phrase = wordList[(int)(Math.random() * wordList.length)];
```

----

# A Trip to Objectville

#### Objects
*   An **object** is a thing that exists in the program. Objects have **state** and **behavior**.
*   **State** is represented by **instance variables**. Instance variables hold data, and each object can have unique values for its instance variables.
*   **Behavior** is represented by **methods**. Methods operate on an object's data.
    *   For example, a `Song` object might have `title` and `artist` as instance variables (state) and methods like `setTitle()` and `play()` (behavior).
*   Objects are created from a **class**.

#### Classes
*   A **class** is a **blueprint** for creating objects. A class defines what data an object will hold (instance variables), and what methods an object will have.
*   A class has one or more **methods**.
*   A source code file (with the `.java` extension) typically holds one class definition. The class definition must be within a pair of curly braces.

#### Global variables or methods
*   There is **no concept of "global" variables or methods** in a Java object-oriented program. In Java, everything must reside within a class.
*   While true global variables don't exist, a method or constant can be made to behave similarly to a global entity by using the `public` and `static` keywords.
    *   A `public static` method can be called from anywhere in an application.
    *   A `public`, `static`, and `final` variable essentially creates a globally available constant.

#### Methods
*   Methods hold instructions for how an object should behave.
*   A method has a name, and a body (the code within the method).
*   A method is declared within the curly braces of a class.
*   You can send things to a method (parameters), and get things back from a method (return types).

#### Naming
*  You can name a class, method, or variable according to the following rules:
  *  It must start with a letter, underscore (_), or dollar sign ($). You can’t start a name with a number.
  *  After the first character, you can use numbers as well. Just don’t start it with a number.
  *  It can be anything you like, subject to those two rules, just so long as it isn’t one of Java’s reserved words.

#### Creating Objects
*   Objects are created using the `new` keyword followed by the class name and parentheses. For example: `Dog d = new Dog();` creates a new `Dog` object.
*   The `new` keyword allocates memory for the new object and calls the constructor.

#### Using Objects
*   The **dot operator** `.` is used to access an object's instance variables and methods.
    *   For example: `d.bark();` calls the `bark` method of the `Dog` object `d`. Also, `d.size = 40;` sets the size of the dog object `d`.

#### Object References
*   A variable that holds an object is actually a **reference** (a remote control) to the object. It doesn't hold the object itself.
*   Multiple reference variables can refer to the **same object**.
*   A reference can be `null`, meaning it does not refer to any object.
*   A reference can be redirected (reprogrammed to refer to a different object), as long as it’s an object of the same class type.
*   If a reference is declared as `final`, it can only ever refer to the same object, and can never be reassigned to another object.

#### Heap Memory
*   When a new object is created using the `new` keyword, the Java Virtual Machine (JVM) allocates memory for that object on the heap. (area of memory where **all objects live**)
*   The size of the memory allocated on the heap for an object depends on the amount of memory the object needs. For example, an object with 15 instance variables would likely need more space than an object with only two instance variables.
*   The Java heap is specifically called the **Garbage-Collectible Heap**, because the memory allocated there is managed by Java's garbage collection process.
*   When an object is no longer needed, the garbage collector reclaims the memory that object was using.
*   An object becomes **eligible for garbage collection** when the JVM determines that the object can no longer be accessed or used by the program.
*   The garbage collector runs periodically, especially when the program is low on memory. The garbage collector **frees up the memory** that was used by unreachable objects, so that memory can be reused. The process is **automatic and invisible** to the programmer.

----

# Know Your Variables

#### Declaring Variables
*  All variables must be declared with a **type** and a **name**.
    *  The **type** of a variable determines the kind of value it can hold (for example, an integer, a character, or a reference to an object).
    *  The **name** of a variable is used in code to access its value.
*  Java **cares about the type** of variables. You cannot put a value of one type into a variable of another, incompatible type. For example, you cannot put a `Giraffe` reference in a `Rabbit` variable.

#### Primitive Types
*  Java has **eight primitive types**: `boolean`, `char`, `byte`, `short`, `int`, `long`, `float`, and `double`.
*  These primitive types are used for storing simple values, not objects.
*  Each primitive type has a specific size and range of values it can represent.

```java
long big = 3456789L;  // notice L here for long
float f = 32.5f;      // notice f here for float
```

#### Reference Variables
*   There is actually no such thing as an object variable. There’s only an object reference variable.
*   A reference variable holds a **reference** (or "remote control") to an **object**, not the object itself.
*   Think of a reference variable as a remote control to an object, not the object itself.
*   The reference variable can be programmed to refer to different objects of the same type during runtime unless declared as final. If a reference variable is marked `final`, it can only ever refer to the same object.
*   Multiple references can refer to the same object.
*   How big is reference variable? Don't know. Dependent on JVM implementation. All references for a given JVM will be the same size regardless of the objects they reference, but each JVM might have a different way of representing references, so references on one JVM may be smaller or larger than references on another JVM.

#### Object References
*   When you declare a reference variable, you must specify the type of the object it can refer to. For example, a `Dog` reference can only refer to `Dog` objects.
*   A reference can be set to `null`, which means it does not refer to any object.
*   A reference can be redirected to a different object of the same class type.
*   **A reference variable's type does not determine the type of the object**, but rather the type of object that a reference variable can refer to.
    *   For example, `Animal a = new Dog();` is valid because `Dog` is a type of `Animal` but a reference variable of type `Animal` called `a` is still referring to a `Dog` object.

#### Arrays
*   Arrays are always objects, whether they’re declared to hold primitives or object references. Think of them as tray of cups. Now cups can either store the primitive itself or a remote control (reference) to object

----

# How Objects Behave

#### Instance Variables
*   Each **object** (instance of a class) can have its own unique values for its instance variables.
*   Instance variables represent the **state** of an object.
*   Instance variables should be marked **private**, to implement encapsulation.

#### Methods
*   **Methods** operate on the data (state) of an object.
*   Methods use and can change the values of an object's **instance variables.**
*   Methods can take **arguments** (also called **parameters**) that provide input to the method.
*   Methods can also **return** a value as the result of their operation.
*   Methods are declared within the curly braces of the class, and have a method signature (name and arguments) and a body.

#### Encapsulation
*   **Encapsulation** is the practice of keeping instance variables **private** and providing controlled access to them through public **getter** and **setter** methods.
*   **Getters** are public methods that return the value of a private instance variable.
*   **Setters** are public methods that allow you to change the value of a private instance variable.
*   Encapsulation **protects the state** of an object from uncontrolled changes.
*   Using **private instance variables** provides more flexibility in how data is handled and allows for future changes without breaking the program.

#### Method Arguments (Parameters)
*   Methods can accept one or more arguments (parameters).
*   When a method is called, the values passed as arguments are copied to the method's parameter variables (a concept called "pass by value").
*   Java is pass-by-value
    *   **Primitive types** are passed by value, meaning that a copy of the value is passed to the method. Changing the value of the parameter in the method will not change the original variable.
    *   **Reference types** are also passed by value. However, the value that is copied is the object's reference, not the object itself. So, if the method modifies the object using the copied reference, the original object outside of the method will be modified.

#### Method Return Types
*   Methods can have a **return type**, which specifies the type of value the method will return.
*   If a method declares a non-void return type, it must return a value compatible with the declared return type.
*   If a method does not return a value, its return type is declared as `void`.
*   When a method returns a primitive value, a copy of the value is returned to the caller.
*   When a method returns a reference value, the caller receives a copy of the reference, not a copy of the object.

#### Default values
*   Local variables do NOT get a default value! The compiler complains if you try to use a local variable before the variable is initialized.
*   Instance variables always get a default value. If you don’t explicitly assign a value to an instance variable or you don’t call a setter method, the instance variable still has a value!

```
booleans false
characters 0
integers 0
floating points 0.0
references null
```

#### Comparing primitives and reference types
*   Use == to compare two primitives or to see if two references refer to the same object. 
*   Use the equals() method to see if two different objects are equal. 

----

# Extra Strength Methods

#### New Java Concepts

*   **For Loops**:
    *   Structure: **initialization**, **boolean test**, **iteration expression**.
        *   **Initialization**: Executes once at loop start.
        *   **Boolean test**: Evaluated before each iteration.
        *   **Iteration expression**: Executes at end of each loop.
    *   **Enhanced for loop**: Simplifies iteration through arrays and collections implementing the `Iterable` interface.
        ```java
        int[] locations = {1, 2, 5};
        for (int location: locations) {
        }
        ```
*   **Post-increment Operator (`++`)**:
    *   Increments a variable *after* its value is used.
*  **Break Statement**:
    *   Exits a loop prematurely.
*   `ArrayList` **vs. Regular Arrays**:
    *  Regular arrays have fixed size. `ArrayList` objects have a dynamic size.
    *  Regular arrays you access only `length` instance variable, for `ArrayList` we can call so many methods on it (see below)

# Using the Java Library

#### ArrayList

*   **`ArrayList`** is a class that implements the **`List` interface**, which is part of the Java Collections Framework.
*   `ArrayList` is a dynamic array, which allows it to grow and shrink as needed.
*   Common `ArrayList` methods:
    *   `add(element)`: Adds an element to the end of the list.
    *   `add(index, element)`: Inserts an element at a specific index.
    *   `remove(index)`: Removes an element at a specific index.
    *   `remove(Object o)`: Removes first occurrence of the specified element
    *   `get(index)`: Retrieves the element at a specific index.
    *   `size()`: Returns the number of elements in the list.
    *   `contains(element)`: Checks if the element exists in the list.
*   `ArrayList` objects can only hold **objects**, not primitives. Primitive values can be wrapped as objects using wrapper classes.

```java
ArrayList<Egg> myList = new ArrayList<Egg>();
Egg egg1 = new Egg();
myList.add(egg1);
Egg egg2 = new Egg();
myList.add(egg2);

boolean isIn = myList.contains(egg1);
int theSize = myList.size();
int idx = myList.indexOf(egg2);
boolean empty = myList.isEmpty();
myList.remove(egg1);
```

#### Java API Documentation

*   The **Java API documentation** provides comprehensive information about classes, methods, and packages.
*   Navigating the API docs:
    *   **Top down**: Find a package and drill down.
    *   **Class-first**: Find the class and click it.
    *   **Search**: Use search to go to a specific method, class, package or module.

#### Packages

*   Packages are used to
    *   **organize** related classes and interfaces into a hierarchical namespace.
    *   prevent **naming conflicts** when multiple classes with the same name exist.
    *   provide a level of security, because you can restrict the code you write so that only other classes in the same package can access it.

```java
// javax.swing

// java.util
ArrayList

// java.lang -- this is sort of pre-imported for free, hence we don't have to import this.
System (System.out.println)
Math
String

```

```java
// Either you add import statement and then you can use class name directly
// NOTE: import does not actually make the imported class code to come to class that's importing it. It's not like C++'s #include
//       Just saves you from typing, that's it. So don't have to worry about code getting bloated. 
import java.util.ArrayList;
ArrayList<Egg> eggs = ...

// Or you use the full name explicitly everywhere in the code.
java.util.ArrayList<Egg> eggs = ...
```

# Better living in Objectville

#### Inheritance

*   **Inheritance** allows a class (subclass) to inherit **members** (instance variables and methods) of the superclass.
    *   Example: `Animal` superclass with `Feline` and `Canine` subclasses.
*   The **`extends`** keyword is used to establish an inheritance relationship in Java.
*   Subclasses can:
    *   Add new methods and instance variables.
    *   **Override** inherited methods. We do not override inherited instance variables, but rather redefine them (if required, though it's almost not required)
*   **IS-A test**: Use the IS-A test to check if the inheritance hierarchy makes sense (e.g., a Cat IS-A Pet).
    *   works anywhere in the inheritance tree. If your inheritance tree is well-designed, the IS-A test should make sense when you ask any subclass if it IS-A any of its supertypes. Example: Wolf --> Canine --> Animal (here Wolf IS-A Canine, and also Wolf IS-A Animal)
    *   means that subclass can do anything that superclass can do + subclass can more things specific to that subclass.
    *   works one-directionally. Wolf IS-A Animal, but reverse is not true.
*   You can call a method on an object reference only if the class of the reference type actually has the method.
*   When a method is invoked, it will use the object type's implementation of that method. You’re calling the most specific version of the method for that object type. **In other words, the lowest one wins!**.
*   **HAS-A test**: There is a HAS-A test too, which is effectively composition (Bathroom HAS-A Tub means that Bathroom has a Tub instance variable)

```java
@Override
public void roam() {
 super.roam();          // calls inherited version of roam()
 // my own roam stuff
}

@Override
public void eat() {
 // my own eat stuff, don't call the superclass's eat()
}
```

#### Rules of Overriding
*   The argument list (parameter types) of the overriding method must match **exactly** with the arguments of the overridden method.
*   The return type of the overriding method must be **compatible** with the return type of the overridden method.
*   The overriding method cannot be less accessible than the overridden method.
    *   A `public` method cannot be overridden to be `private`.

#### Rules of Overloading
*   An overloaded method is just a different method that happens to have the same method name. It has nothing to do with inheritance and polymorphism. An overloaded method is NOT the same as an overridden method.
    *  You can’t change ONLY the return type. You must change the argument list
    *  The return types can be different.
    *  You can vary the access levels in any direction.

#### Polymorphism  
```java
Animal[] animals = new Animal[5];
animals[0] = new Dog();
animals[1] = new Cat();
animals[2] = new Wolf();
animals[3] = new Hippo();
animals[4] = new Lion();

for (Animal animal : animals) {
 animal.eat();
 animal.roam();
}
```

```java
class Vet {
 // can take any class which IS-A Animal (say Canine, or Wolf)
 public void giveShot(Animal a) {
 a.makeNoise();
 }
}
```

The three things that can prevent a class from being subclassed are
*   **Access Control**: A **non-public** class (a class that does not have the `public` keyword modifier) can only be subclassed by classes within the same package. Classes in different packages will not be able to subclass it.
*   **`final` Keyword**: If a class is declared with the `final` modifier, it cannot be extended. A `final` class is at the end of the inheritance line, and no other class can inherit from it.
*   **Private Constructors**: A class with only **private constructors** cannot be subclassed. Private constructors prevent the class from being instantiated by code outside the class, thus preventing subclassing.

# Serious Polymorphism

#### Abstract Classes

*   **Abstract classes** cannot be instantiated.
*   Abstract classes can have both **abstract methods** and **concrete methods**. Abstract methods must be implemented by concrete subclasses.
*   Abstract classes can be thought of as **partially implemented classes**.
*   If you declare an abstract method, you MUST mark the class abstract as well. You can’t have an abstract method in a non-abstract class. Abstract methods exist solely for polymorphism purpose. In subclass you eventually override with similar conditions (same exact arg list, compatible return type and wider or same access modifier).

```java
abstract class Animal {
  abstract void eat();
  abstract void roam();
}

abstract class Canine extends Animal {
  abstract void cleanCanineTooth();   // must be overridden in the concrete class extending Canine;

  void sharpenTooth() {        // need not be overridden in Canine's subclass as we have provided impl here
    // random implementation
  }

  @Override
  void eat() {    // we can implement eat here only so that Canine's subclass (say Wolf) don't have to.

  }
}

Canine c;
c = new Dog(); // Fine. Abstract class can be used as reference variable, which can point to an object of a concrete class.
c = new Canine(); // Not fine, compilation error.
```

#### Interfaces

*   **Interfaces** are a way to achieve **polymorphism** by defining a **contract** specifying what methods a class must implement.
*   Interfaces declare methods that implementing classes must provide.
*   **A class can implement multiple interfaces**, using the keyword `implements`.
*   Interfaces do not have method implementations, only method signatures. After Java 8 however, interfaces's methods too can have impl, the method just needs to be marked `default`.
*   Interfaces can be thought of as **"pure abstract classes"**.
*   A Java class can have only one parent (superclass), and that parent class defines who you are. But you can implement multiple interfaces, and those interfaces defines role that you can play.

| Feature                      | Abstract Class                  | Interface                       |
|------------------------------|----------------------------------|---------------------------------|
| Multiple inheritance         | Not supported                   | Supported                       |
| Constructors                 | Allowed (but still can't instantiate this class)                        | Not allowed                     |
| Fields                       | Any type                        | Only `public static final`      |
| Method implementation        | Allowed                         | Allowed (default, static)       |
| Relationship                 | "Is-a" relationship             | "Can-do" relationship like Serializable, Flyable         |

#### `Object` Class

*   The `Object` (java.lang.Object) class is the root of all classes in Java.
*   Any class that doesn’t explicitly extend another class, implicitly extends `Object`
*   `Object` class has some basic methods available to all classes, such as (there are more):
    *   `equals(Object o)`: Used to check if two objects are equal.
    *   `getClass()`: Returns the class object of an object.
    *   `toString()`: Returns a string representation of the object.
    *   `hashCode()`: Returns an integer hash code value for the object.
*   `Object` is a non-abstract class, so you can make object of `Object` class.

```java
ArrayList<Object> myDogArrayList = new ArrayList<Object>();
Dog aDog = new Dog();
myDogArrayList.add(aDog);

Dog d = myDogArrayList.get(0);    // won't compile, because get here returns type Object

Object d = myDogArrayList.get(0); // works
d.bark();                         // but this won't work, hence `d` is not of much use.

Cat c = (Cat) d;                ` // runtime ClassCastException.

if (d instanceof Dog) {
  Dog x = (Dog) d;                // explicit casting
  x.bark();                       // works now.
}
```

* **Compiler checks the class of the reference variable, not the class of the actual object at the other end of the reference.**

* Now say we want few animals (Cat and Dog) to have pet behaviors too
  * Option 1: Put pet methods (either as abstract or concrete) in Animal -- bad because many non-petable Animal (like lion) also inherit these pet methods then
  * Option 2: Create another abstract class Pet, and only Dog and Cat extend Pet too (alongside Animal). Deadly diamond of death problem here.
  * Option 3: Pet as interface.

#### **Diamond Problem in Java (Superclass and Interface)**
##### Conflict between superclass and interface
* Scenario
  - **Class B**: Has an implemented method `foo()`.  
  - **Interface X**: Has a default method `foo()`.  
  - **Class A**: Extends `Class B` and implements `Interface X`.

* Key Resolution Rule
  - **Superclass Takes Precedence**:  If a method is present in both a superclass and an interface (as a default method), the superclass implementation is used automatically.  

* **Example**
  ```java
  class B {
      void foo() {
          System.out.println("B's foo");
      }
  }
  
  interface X {
      default void foo() {
          System.out.println("X's foo");
      }
  }
  
  class A extends B implements X {
      // No override needed; B's foo() is used.
  }

  A a = new A();
  a.foo(); // Output: "B's foo"
  ```

* Explicit Override
  - To explicitly use the interface's default method, override `foo()` in `Class A` and call `X.super.foo()`:

  ```java
  class A extends B implements X {
      @Override
      public void foo() {
          X.super.foo(); // Calls X's foo
      }
  }
  A a = new A();
  a.foo(); // Output: "X's foo"
  ```

##### Conflict between multiple interfaces
* If two interfaces have conflicting default methods, the implementing class must explicitly resolve the conflict or it won't compile.
  ```java
  interface Y { default void foo() { System.out.println("Y's foo"); } }
  interface Z { default void foo() { System.out.println("Z's foo"); } }
  
  class A implements Y, Z {
      @Override
      public void foo() {
          Y.super.foo(); // Resolve ambiguity
      }
  }
  A a = new A();
  a.foo(); // Output: "Y's foo"
  ``` 
