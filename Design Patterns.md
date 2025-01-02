Appendix:
* Dive Into Design Patterns by Alexander Shvets (aka https://refactoring.guru/design-patterns)
* Head First Design Patterns (2nd edition)

Index:
* [Prerequisite comcepts](#prerequisite-concepts)
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
* [Behavioral Patterns](behavioral-patterns)
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

# Prerequisite comcepts

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
| **Aggregation**  | "Has-a" (shared)    | Independent                  | University ↔ Departments            | ` University <>----- Department ` (unfilled diamond with solid line) |
| **Composition**  | "Owns-a" (exclusive)| Part depends on the whole    | Car ↔ Engine                        | ` Car ◆----- Engine ` (filled diamond with solid line) |

#### Introduction to Design Patterns

*   **Design patterns** are common solutions to recurring problems in software design. They are like customizable blueprints.
*   A pattern is like a blueprint, showing the result, features, and a flexible order of implementation.
*   A typical pattern description includes the **Intent**, **Motivation**, **Structure**, and a **Code Example**.
*   Design patterns can be categorized by:
    *   **Complexity**, **level of detail**, and **scale** of applicability.
    *   **Idioms**, which are low-level and language-specific patterns.
    *   **Architectural patterns**, which are high-level and apply to entire applications.
    *   **Creational patterns** that handle object creation.
    *    **Structural patterns** that focus on how objects are assembled into larger structures.
   *  **Behavioral patterns** which address communication and responsibility between objects.
*   Design patterns are not invented concepts; rather they are discovered solutions to recurring problems.
*   The concept of patterns was first described by Christopher Alexander, and was later applied to software engineering by the Gang of Four (GoF).

#### Why Learn Patterns?

*   Design patterns offer **tested solutions** for common software design problems.
*   Learning patterns teaches object-oriented design principles.
*   Design patterns provide a **common language** for developers to communicate more efficiently.



*   Creational patterns provide object creation mechanisms that increase flexibility and reuse of existing code.
*   Structural patterns explain how to assemble objects and classes into larger structures, while keeping these structures flexible and efficient.
*   Behavioral patterns take care of effective communication and the assignment of responsibilities between objects.
