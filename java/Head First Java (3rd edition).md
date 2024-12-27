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

*   Use == to compare two primitives or to see if two references refer to the same object. 
*   Use the equals() method to see if two different objects are equal. 

