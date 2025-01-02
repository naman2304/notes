Appendix:
* Dive Into Design Patterns by Alexander Shvets (aka https://refactoring.guru/design-patterns)
* Head First Design Patterns (2nd edition)

Index:
* [Prerequisite concepts](#prerequisite-concepts)
* [Creational Patterns](#creational-patterns)
  * [Factory method](#factory-method)  
  * [Abstract Factory](#abstract-factory)  
  * [Builder](#builder)  
  * [Prototype](#prototype)  
  * [Singleton](#singleton)  
* [Structural Patterns](#structural-patterns)
  * [Adapter](#adapter)  
  * [Bridge](#bridge)  
  * [Composite](#composite)  
  * [Decorator](#decorator)  
  * [Facade](#facade)  
  * [Flyweight](#flyweight)  
  * [Proxy](#proxy)  
* [Behavioral Patterns](#behavioral-patterns)
  * [Chain of Responsibility](#chain-of-responsibility)  
  * [Command](#command)  
  * [Iterator](#iterator)  
  * [Mediator](#mediator)  
  * [Memento](#memento)  
  * [Observer](#observer)  
  * [State](#state)  
  * [Strategy](#strategy)  
  * [Template Method](#template-method)  
  * [Visitor](#visitor)  

# Prerequisite concepts

#### Basics of OOP

*   **Object-oriented programming (OOP)** is a paradigm that focuses on bundling **data** and the **behavior** related to that data into **objects**.
*   **Classes** are **blueprints** for creating objects and they define the structure of objects.
*   A **class** defines **fields** (attributes or data) and **methods** (behaviors) that are the **members** of the class.
*   An object's **state** refers to the data stored in its fields, and the **behavior** is defined by its methods.
*   **Objects** are **instances** of classes, and multiple objects can be created from the same class with different attribute values.

<a href="url"><img src="/metadata/didp_basicsofoop.png" width="480" ></a>

#### Class Hierarchies

*   Classes can be arranged into **hierarchies** where **subclasses** inherit from **superclasses** (or parent classes).
*   A **superclass** or **base class** defines common attributes and behaviors.
*   **Subclasses** inherit the state and behavior of their parent classes, and can add unique attributes or behaviors or **override** methods.
*   **Method overriding** allows subclasses to change or enhance the default behavior of inherited methods.
*   UML diagrams of class hierarchies can be simplified to emphasize the **relationships** between classes, rather than the details of each class.

<a href="url"><img src="/metadata/didp_classheirarchies.png" width="480" ></a>

#### Pillars of OOP

*   OOP is based on four key principles: **Abstraction**, **Encapsulation**, **Inheritance**, and **Polymorphism**.
*   **Abstraction** is modeling real-world objects with only the **relevant details** for a specific context, ignoring irrelevant details. Abstraction is context-specific. Like a plane in flight simulation context may have different attributes + behavior (like speed, altitude), but in booking seats context may have different attributes + behavior (like seatMap)
*   **Encapsulation** hides the internal implementation of an object, and exposes only a **public interface** for interactions with other objects.
*   **Inheritance** enables the creation of new classes based on existing ones, allowing code reuse. Subclasses inherit the same interface and must implement all abstract methods, even if not applicable. A subclass can only extend one superclass, but can implement multiple interfaces.
*   **Polymorphism** allows subclasses to override base methods of a superclass, so that each subclass can exhibit its specific behavior.

#### Relations between objects
*   **Association**
    *   is a general relationship between two objects where they are aware of each other and can interact.
    *   "Knows-a" relationship.
    *   Can be of two types: aggregation or composition.
    *   In general, you use an association to represent something like a field in a class. The link is always there, in that you can always ask an order for its customer
    *   Example: A Professor teaches Students. Both Professor and Students can exist independently.
    ```java
    class Professor {
        String name;
        List<Student> students; // Association
    }
    
    class Student {
        String name;
    }
    ```
*   **Dependency**
    *   is a relationship where one object uses another temporarily to perform some operation.
    *   "Uses-a" relationship.
    *   Example: A Car depends on a FuelStation to refuel, but it doesn’t own it or retain a reference to it.
    ```java
    class Car {
        void refuel(FuelStation station) {
            station.provideFuel();
        }
    }
    
    class FuelStation {
        void provideFuel() {
            System.out.println("Fuel provided!");
        }
    }
    ```
*   **Aggregation**
    *    A type of association where one object is a "whole" that groups other objects as "parts", but the parts can exist independently of the whole.
    *    "Has-a" relationship with **shared ownership**
    *    Example: A University aggregates Departments, but departments can exist even if the university is closed.
    ```java
    class University {
        List<Department> departments; // Aggregation
    }
    
    class Department {
        String name;
    }
    ```
*   **Composition**
    *    A type of association where one object is a "whole" that owns its "parts," and the parts cannot exist independently of the whole.
    *    "Owns-a" relationship with **exclusive ownership**.
    *    A Car is composed of an Engine. If the car is destroyed, the engine no longer exists.
    ```java
    class Car {
        Engine engine; // Composition
        Car() {
            engine = new Engine();
        }
    }
    
    class Engine {
        String type;
    }
    ```

| **Concept**      | **Key Idea**        | **Lifetime**                 | **Example**                         | **UML**                 |
|------------------|---------------------|------------------------------|-------------------------------------| ----------------------- |
| **Association**  | "Knows-a"           | Independent                  | Teacher ↔ Students                  | ` Professor → Student ` (solid arrow) |
| **Dependency**   | "Uses-a"            | Short-lived (temporary)      | Car ↔ FuelStation                   | ` Car  ---> FuelStation ` (dotted arrow) |
| **Aggregation**  | "Has-a" (shared)    | Independent                  | University ↔ Departments            | ` University <>-----> Department ` (unfilled diamond with solid arrow) (yes it looks dotted, but consider it solid) |
| **Composition**  | "Owns-a" (exclusive)| Part depends on the whole    | Car ↔ Engine                        | ` Car ◆-----> Engine ` (filled diamond with solid arrow) (yes it looks dotted, but consider it solid) |

#### Introduction to Design Patterns

*   **Design patterns** are common solutions to recurring problems in software design. They are like customizable blueprints.
*   A pattern is like a blueprint, showing the result, features, and a flexible order of implementation.
*   A typical pattern description includes the **Intent**, **Motivation**, **Structure**, and a **Code Example**.
*   Design patterns can be categorized by:
    *   **Creational patterns** provide object creation mechanisms that increase flexibility and reuse of existing code.
    *   **Structural patterns** explain how to assemble objects and classes into larger structures, while keeping these structures flexible and efficient.
    *   **Behavioral patterns** take care of effective communication and the assignment of responsibilities between objects.


#### Features of Good Design

*   **Code Reuse**
    *   **Reduces development costs** by avoiding redundant work and saving time.
    *   Allows for **faster time to market**, which gives a competitive advantage.
    *   Promotes more efficient and broader reach to potential customers because it lowers costs of development and allows for more money to be spent on marketing.
    *   Usually there are 3 levels of reuse
        *   Lowest level is reusing a class
        *   Middle level is design patterns
        *   Highest level is framework (like JUnit), common theme here is "don't call us, let us call you" -- like JUnit executes the tests.
    *   Good design anticipates future changes and enables code reuse by **extracting common functionality into separate modules or classes**.
    *   Changes to an application should not affect the whole program.
    *   Code reuse enhances efficiency and **reduces the need for redundant code**.

*   **Flexibility and Adaptability**
    *   Good software design accounts for future changes in requirements.
    *   **Encapsulating what varies** is a key principle to minimize the impact of changes in software.
    *   Changes should be isolated to specific modules or classes, **preventing widespread issues**.
    *   Software must be able to adapt to new features, changes requested by clients, and changes in the environment.
    *   When changes occur, the program should not break and should be able to adapt to the change easily.

#### Design Principles

*   **Encapsulate What Varies**
    *   This principle involves identifying the aspects of an application that change and separating them from what stays the same.
    *   The main goal is to **minimize the effect caused by changes**.
    *   Isolating the parts of a program that vary into independent modules **protects the rest of the code from adverse effects**.
    *   This principle can be applied at both the method level and class level.
        *   At the **method level**, it involves extracting complex behavior to reduce the complexity and blur the primary responsibility of the method.
        *   At the **class level**, it involves extracting related behaviors, fields, and methods to a new class when they blur the primary responsibility of the containing class.

*   **Program to an Interface, not an Implementation**
    *    Depend on abstractions, not on concrete classes.
 
    <a href="url"><img src="/metadata/didp_programtointerface.png" width="480" ></a>

*   **Favor Composition Over Inheritance**. Problems with inheritance:
    *   A subclass can’t reduce the interface of the superclass. You have to implement all abstract methods of the parent class even if you won’t be using them.
    *   When overriding methods you need to make sure that the new behavior is compatible with the base one. It’s important because objects of the subclass may be passed to any code that expects objects of the superclass and you don’t want that code to break.
    *   Inheritance breaks encapsulation of the superclass because the internal details of the parent class become available to the subclass
    *   Subclasses are tightly coupled to superclasses. Any change in a superclass may break the functionality of subclasses.
    *   Trying to reuse code through inheritance can lead to creating parallel inheritance hierarchies. Inheritance usually takes place in a single dimension. But whenever there are two or more dimensions, you have to create lots of class combinations, bloating the class hierarchy to a ridiculous size.
    *   Example: right hand side is actually Strategy Pattern.

    <a href="url"><img src="/metadata/didp_inbadcompgood_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_inbadcompgood_1.png" width="480" ></a>

#### SOLID Principles

*   **Single Responsibility Principle (SRP)**
    *   A class should have only **one reason to change**.
    *   Each class should be responsible for a single part of the functionality provided by the software.
    *   This principle aims to reduce complexity.
    *   By encapsulating a single responsibility within a class, the code is more focused and maintainable.

    <a href="url"><img src="/metadata/didp_solid1_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_solid1_1.png" width="480" ></a>
    

*   **Open/Closed Principle (OCP)**
    *   Classes should be **open for extension but closed for modification**.
    *   You should be able to extend a class by creating a subclass without modifying its existing code.
    *   If a class is already developed, tested, reviewed, and included in some framework or otherwise used in an app, trying to mess with its code is risky. Instead of changing the code of the class directly, you can create a subclass and override parts of the original class that you want to behave differently. You’ll achieve your goal but also won’t break any existing clients of the original class.
    *   This principle isn’t meant to be applied for all changes to a class. If you know that there’s a bug in the class, just go on and fix it; don’t create a subclass for it.
    *   Classes should have a **well-defined interface** that is stable and will not change in the future.
    *   Example: say we want to add more shipping ways, then we would have to change Order class. Using Strategy pattern we decouple it, and now we can simply add another Shipping way.

    <a href="url"><img src="/metadata/didp_solid2_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_solid2_1.png" width="480" ></a>
    
*   **Liskov Substitution Principle (LSP)**
    *   Subclasses should be substitutable for their base classes **without altering the correctness of the program**.
    *   **Invariants of a superclass must be preserved** by its subclasses. Invariants are conditions in which an object makes sense.
        *   Parameter types in a method of a subclass should match the parameter types in the method of the superclass.
        *   The return type in a method of a subclass should match or be a subtype of the return type in the method of the superclass (inverse of above rule)
        ```java
        class A {
          Cat buyCat() {}
        }

        class B extends A {
          // GOOD, won't break client.
          BengalCat buyCat() {}

          // BAD, will break client -- won't compile.
          Animal buyCat() {}
        }
        ```
        *    A method in a subclass shouldn’t throw types of exceptions which the base method isn’t expected to throw. In other words, types of exceptions should match or be subtypes of the ones that the base method is already able to throw
        *    A subclass shouldn’t strengthen pre-conditions. For example, if a base method has a parameter of type `int`, and a subclass overrides it by requiring the argument to be positive (throwing an exception for negatives), it strengthens the preconditions, breaking client code that previously worked with negative values.
        *    A subclass shouldn’t weaken post-conditions. Say you have a class with a method that works with a database. A method of the class is supposed to always close all opened database connections upon returning a value. You created a subclass and changed it so that database connections remain open so you can reuse them. But the client might not know anything about your intentions.
        *    A subclass shouldn’t change values of private fields of the superclass. Possible via reflection mechanisms.
        ```java
        class Superclass {
            private int value = 42;
        }
        
        class Subclass extends Superclass {
            public void accessPrivateField() throws Exception {
                Field field = Superclass.class.getDeclaredField("value");
                field.setAccessible(true);
                int value = (int) field.get(this);
                System.out.println("Private value: " + value);
            }
        }
        ```


Here, if client doesn't check if document is not of read type and calls save, it will blow up!
    <a href="url"><img src="/metadata/didp_solid3_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_solid3_1.png" width="480" ></a>

*   **Interface Segregation Principle (ISP)**
    *   Clients shouldn't be forced to **depend on methods they do not use**.
    *   Interfaces should be narrow enough so that client classes do not need to implement behaviors they do not require.
    *   This principle aims to **keep classes focused** by avoiding implementing methods that are not relevant.

    <a href="url"><img src="/metadata/didp_solid4_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_solid4_1.png" width="480" ></a>
    
*   **Dependency Inversion Principle (DIP)**
    *   High-level classes shouldn’t depend on low-level classes. Both should depend on abstractions. Abstractions shouldn’t depend on details. Details should depend on abstractions.
    *   The dependency inversion principle often goes along with the open/closed principle (see example there)

    <a href="url"><img src="/metadata/didp_solid5_0.png" width="480" ></a>
    <a href="url"><img src="/metadata/didp_solid5_1.png" width="480" ></a>



# Creational Patterns

# Structural Patterns

# Behavioral Patterns
