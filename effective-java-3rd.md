# Effective Java 3rd Edition Summary

* [Creating and destroying objects](#creating-and-destroying-objects)
* [Methods common to all objects](#methods-common-to-all-objects)
* [Classes and Interfaces](#classes-and-interfaces)
* [Generics](#generics)
* [Enums and annotations](#enums-and-annotations)
* [Lambdas and streams](#lambdas-and-streams)
* [Methods](#methods)
* [General Progamming](#general-programming)
* [Exceptions](#exceptions)
* [Concurrency](#concurrency)
* [Serialization](#serialization)


## Creating and destroying objects

__Item 1 : Static factory methods__

Pros
 - They have a name. So for instantiating classes with complex arguments to constructor, rather use a static method factory with good naming.
 - You can use them to control the number of instance (example : `Boolean.valueOf`). We can cache / precompute them to avoid creating instance of the class when asked one.
 - You can return a subtype of the return class 

Cons
 - You can't subclass a class without public or protected constructor
 - It can be hard to find the static factory method for a new user of the API

Example :
```java
public static Boolean valueOf(boolean b) {
	return b ? Boolean.TRUE :  Boolean.FALSE;
}
```

__Item 2 : Consider a builder pattern__

When few `required` parameters, but too many `optional` parameters, how do we create constructors?

First Way: **Telescoping Pattern** (safe, but not so readable)
 - in which you provide a constructor with only the required parameters, another with a
single optional parameter, a third with two optional parameters, and so on, culminating in a constructor with all the optional parameters.
 - But it's too much to maintain and error prone.

Example :
```java
public class NutritionFacts {
	private final int servingSize; 	// required.
	private final int servings;	// required.
	private final int calories;
	private final int fat;
	private final int sodium;

	public NutritionFacts(int servingSize, int servings) {
		this(servingSize, servings, 0);
	}
	public NutritionFacts(int servingSize, int servings, int calories) {
		this(servingSize, servings, calories, 0);
	}
	public NutritionFacts(int servingSize, int servings, int calories, int fat) {
		this(servingSize, servings, calories, fat, 0);
	}
	public NutritionFacts(int servingSize, int servings, int calories, int fat, int sodium) {
		this.servingSize = servingSize;
		this.servings = servings;
		this.calories = calories;
		this.fat = fat;
		this.sodium = sodium;
	}
}
```

Second Way: **JavaBean Pattern** (unsafe, but readable)
 - Provide setter for all
 - But here, object is in inconsistent state during the various statements when client is trying to setup the object correctly. No atomocity while creation of object -- thread unsafe.
 - Also, it prevents the possibility of making a class immutable.

Example :
```java
public class NutritionFacts {
	private final int servingSize; 	// required.
	private final int servings;	// required.
	private final int calories;
	private final int fat;
	private final int sodium;

	public NutritionFacts() {
	}
	// Setters
	public void setServingSize(int val) { servingSize = val; }
	public void setServings(int val) { servings = val; }
	public void setCalories(int val) { calories = val; }
	public void setFat(int val) { fat = val; }
	public void setSodium(int val) { sodium = val; }
}

NutritionFacts cocaCola = new NutritionFacts();
cocaCola.setServingSize(240);
cocaCola.setServings(8);
cocaCola.setCalories(100);
cocaCola.setSodium(35);
```

Third Way: **Builder Pattern** (safe and readable)
 - It's easier to read and write
 - Client calls a constructor (or static factory) with all of the **required** parameters and gets a builder object. Then the client calls setter-like methods on the builder object to set each optional parameter of interest. Finally, the client calls a parameterless build method to generate the object, which is typically immutable
 - You can prevent inconsistent state of your object as instance creation of main outer class is atomic.
 - Your class can be immutable (instead of using a java bean)
 - Builder is usually static member class of the class it builds (so that client can access Builder directly without having to instantiate outer class)
 - Checks for validity for individual parameters go in Builder's constructor and methods.
 - Checks for multiple parameter's invariants goes to outer class's **private** constructor *after copying parameters from the builder* (Item 50). If a check fails, throw an IllegalArgumentException (Item 72) whose detail message indicates which parameters are invalid (Item 75).

Example :
```java
public class NutritionFacts {
	private final int servingSize;
	private final int servings;
	private final int calories;
	private final int fat;
	private final int sodium;

	public static class Builder {
		//Required parameters
		private final int servingSize:
		private final int servings;
		//Optional parameters - initialized to default values
		private int calories			= 0;
		private int fat 			= 0;
		private int sodium 			= 0;

		public Builder (int servingSize, int servings) {
			this.servingSize = servingSize;
			this.servings = servings;
		}
		public Builder calories (int val) {
			calories = val;
			return this;				
		}
		public Builder fat (int val) {
			fat = val;
			return this;				
		}
		public Builder sodium (int val) {
			sodium = val;
			return this;				
		}
		public NutritionFacts build() {
			return new NutritionFacts(this);
		}
	}
	private NutritionFacts(Builder builder) {
		// Nested classes (inner or static) have full access to the private fields and methods of the enclosing class and vice versa.
		servingSize		= builder.servingSize;
		servings 		= builder.servings;
		calories		= builder.calories;
		fat 			= builder.fat;
		sodium 			= builder.sodium;
	}

NutritionFacts cocaCola = new NutritionFacts.Builder(240, 8).calories(100).sodium(35).carbohydrate(27).build();
}
```

Builder pattern is good for class heirarchies too.
 - Use a parallel hierarchy of builders, each nested in the corresponding class. Abstract classes have abstract builders; concrete classes have concrete builder.
 - This technique, wherein a subclass method (build function) is declared to return a subtype of the return type declared in the superclass, is known as *covariant return typing*. It allows clients to use these builders without the need for casting (fluent API).

```java
public abstract class Pizza {
	public enum Topping { HAM, MUSHROOM, ONION, PEPPER, SAUSAGE };
 	final Set<Topping> toppings;

	// This is generic type with recursive type parameter (Item 30)
	// This, along with the abstract self method, allows method chaining to work properly in subclasses, without the need for casts.
  	public abstract static class Builder<T extends Builder<T>> {
		EnumSet<Topping> toppings = EnumSet.noneOf(Topping.class);

		// Return subclass type T so that chaining happens smoothly
		public T addTopping(Topping topping) {
			toppings.add(Objects.requireNonNull(topping));
			return self();			
		}

		abstract Pizza build();

		// Subclasses must override this method to return "this"
		// Made protected so that only subclass can access / override it, and client code can't call this function.
		protected abstract T self();
	}

	Pizza(Builder<?> builder) {
		// We should always clone the Builder variable to outer class
		// As one Builder can be used to create object, and then changes happen to same Builder to create another object.
		// Hence to keep memory different for variables in these 2 objects, use clone.
		toppings = builder.toppings.clone();
	}
}

public class NYPizza extends Pizza {
	public enum Size { SMALL, MEDIUM, LARGE };
	private final Size size;

	public static class Builder extends Pizza.Builder<Builder> {
		private final Size size;

		public Builder(Size size) {
			this.size = Objects.requireNonNull(size);
		}
		@Override public NyPizza build() {
			return new NYPizza(this);
		}
		@Override protected Builder self() { return this; }
	}

	// Constructor is private.
	private NYPizza(Builder builder) {
		super(builder);
		size = builder.size;
	}
}

public class Calzone extends Pizza {
	private final boolean sauceInside;

	public static class Builder extends Pizza.Builder<Builder> {
		private final boolean sauceInside = false;

		public Builder sauceInside() {
			sauceInside = true;
			return self();
		}
		
		@Override public Calzone build() {
			return new Calzone(this);
		}
		@Override protected Builder self() { return this; }
	}

	// Constructor is private.
	private Calzone(Builder builder) {
		super(builder);
		sauceInside = builder.sauceInside;
	}
}

// See that we are able to call addTopping twice (or even more),
// and it will aggregate various values to a single field `toppings`.
NyPizza pizza = new NyPizza.Builder(SMALL).addTopping(SAUSAGE).addTopping(ONION).build();
Calzone calzone = new Calzone.Builder().addTopping(HAM).sauceInside().build();
```

__Item 3 : Think of Enum to implement the Singleton pattern__


Example :
```java
public enum Elvis() {
	INSTANCE;
	...
	public void singASong(){...}
}
```

__Item 4 : Enforce noninstantiability with a private constructor__

A utility class with only static fields and static methods should never be instantiated.
 - Attempting to enforce noninstantiability by making a class abstract does not work. The class can be subclassed and the subclass instantiated. Furthermore, it misleads the user into thinking the class was designed for inheritance
 - Also, an abstract class can NEVER be `final`, so you can't prevent instantiation. 
 - Hence, create a private constructor to prevent the construction of a useless object.
 - As a side effect, this idiom also prevents the class from being subclassed as all constructors must invoke a superclass constructor, explicitly or implicitly, and here the subclass would have no accessible superclass constructor to invoke.

Example :
```java
public class UtilityClass {
	// Suppress default constructor for non-instantiability
	private UtilityClass(){
		// Not required, but it provides insurance in case the constructor is accidentally invoked from within the class.
		throw new AssertionError();
	}
	...
}
```

__Item 5 : Prefer Dependency Injection to hardwiring resources__

* A common mistake is the use of a singleton or a static utility class for a class that depends on underlying resources.
* The use of dependency injection gives us more flexibility, testability and reusability
* Pass the resources, or factories to create them, into the constructor or static factory or builder

Example : 
```java
public class SpellChecker {
	private final Lexicon dictionary;
	public SpellChecker (Lexicon dictionary) {
		this.dictionary = Objects.requireNonNull(dictionary);
	}
	...
}
```
Example of providing a factory (which will create the resource i.e. type or subtypes of tile)
```java
Mosaic create(Supplier<? extends Tile> tileFactory) { ... }
```

__Item 6 : Avoid creating unnecessary objects__

When possible use the static factory method instead of constructor (Example : Boolean)
Be vigilant on autoboxing. The use of the primitive and his boxed primitive type can be harmful. Most of the time use primitives.

__Item 7 : Eliminate obsolete object references__

Memory leaks can happen in  :
 - A class that managed its own memory
 - Caching objects
 - The use of listeners and callback

In those three cases the programmer needs to think about nulling object references that are known to be obsolete

Example : 
In a stack implementation, the pop method could be implemented this way :

```java
public pop() {
	if (size == 0) {
		throw new EmptyStackException();
	}
	Object result = elements[--size];
	elements[size] = null; // Eliminate obsolete references.
	return result;
}
```

__Item 8 : Avoid finalizers and cleaners__

Finalizers and cleaners are not guaranteed to be executed. It depends on the garbage collector and it can be executed long after the object is not referenced anymore.
If you need to let go of resources, think about implementing the *AutoCloseable* interface.

__Item 9 : Try with resources__

When using try-finally blocks exceptions can occur in both the try and finally block. It results in non clear stacktraces.
Always use try with resources instead of try-finally. It's clearer and the exceptions that can occured will be clearer.

Example :
```java
static void copy(String src, String dst) throws IOException {
	try (InputStream in = new InputStream(src); 
		OutputStream out = new FileOutputStream(dst)) {
		byte[] buf = new byte[BUFFER_SIZE];
		int n;
		while ((n = in.read(buf)) >= 0) {
			out.write(buf, 0, n);
		}
	}
}
```

## Methods common to all objects

__Item 10 : equals__

The equals method needs to be overriden  when the class has a notion of logical equality.
This is generally the case for value classes.

The equals method must be :
 - Reflexive (x = x)
 - Symmetric (x = y => y = x)
 - Transitive (x = y and y = z => x = z)
 - Consistent
 - For non null x, x.equals(null) should return false
 
Not respecting those rules will have impact on the use of List, Set or Map.

__Item 11 : hashCode__

The hashCode method needs to be overriden if the equals method is overriden.

Here is the contract of the hashCode method :
 - hashCode needs to be consistent
 - if a.equals(b) is true then a.hashCode() == b.hashCode()
 - if a.equals(b) is false then a.hashCode() doesn't have to be different of b.hashCode()  
 
If you don't respect this contract, HashMap or HashSet will behave erratically.

__Item 12 : toString__

Override toString in every instantiable classes unless a superclass already did it.
Most of the time it helps when debugging.
It needs to be a full representation of the object and every information contained in the toString representation should be accessible in some other way in order to avoid programmers to parse the String representation.

__Item 13 : clone__

When you implement Cloneable, you should also override clone with a public method whose return type is the class itself.
This method should start by calling super.clone and then also clone all the mutable objects of your class.

Also, when you need to provide a way to copy classes, you can think first of copy constructor or copy factory except for arrays.

__Item 14 : Implementing Comparable__

If you have a value class with an obvious natural ordering, you should implement Comparable.

Here is the contract of the compareTo method : 
 - signum(x.compareTo(y)) == -signum(y.compareTo(x))
 - x.compareTo(y) > 0 && y.compareTo(z) > 0 => x.compareTo(z) > 0
 - x.compareTo(y) == 0 => signum(x.compareTo(z)) == signum(y.compareTo(z))
 
It's also recommended that (x.compareTo(y) == 0) == x.equals(y).
If it's not, it has to be documented that the natural ordering of this class is inconsistent with equals.

When confronted to different types of Object, compareTo can throw ClassCastException.

## Classes and Interfaces

__Item 15 : Minimize the accessibility of classes and members__

Make accessibility as low as possible. Work on a public API that you want to expose and try not to give access to implementation details. Encapsulation.
 - this allows components to be developed, tested, optimized, used, understood, and modified in isolation
 - For top-level (non-nested) classes and interfaces, there are only two possible access levels: package-private and public.
 - If a package-private top-level class or interface is used by only one class, consider making the top-level class a private static nested class of the sole class that uses it
 - For members (fields, methods, nested classes, and nested interfaces), there are 4 possible access levels
   - private: member is accessible only from the top-level class where it is declared
   - package-private: member is accessible from any class in the package where it is declared (if we don't specify accessor, this is what we get -- except for interface members, which are public by default)
   - protected: member is accessible from subclasses of the class where it is declared + from any class in the package where it is declared
   - public: member is accessible from anywhere.
 - Both private and package-private members are part of class's implementation and do not normally impact its exported API. However, can leak into exported API if class implements `Serializable` interface (Item 86 and 87).
 - A `protected` member (like `public` member) is part of the class’s exported API and must be supported forever
 - If a method overrides a superclass method, it cannot have a more restrictive access level in the subclass than in the superclass (Liskov substitution principle. Item 10)
 - Instance fields of public classes should not be public
   - except: can make `static final` fields public only if it is either primitive or reference to immutable object, in which case these are effectively constants (CAPITAL_DECLARED). DO NOT make reference to mutable objects public, even if the field is final (in which case the reference cannot be modified, but the referenced object can be modified).
   - It is wrong for a class to have a public static final array field, as nonzero length arrays are mutable

```java
// SECURITY HOLE!
public static final Thing[] VALUES = { ... };

// Way 1 to fix: can make the public array private and add a public immutable list
private static final Thing[] PRIVATE_VALUES = { ... };
public static final List<Thing> VALUES =
 Collections.unmodifiableList(Arrays.asList(PRIVATE_VALUES));

// Way 2 to fix: can make the array private and add a public method that returns a copy of a private array
private static final Thing[] PRIVATE_VALUES = { ... };
public static final Thing[] values() {
	return PRIVATE_VALUES.clone();
}
```

 - It is acceptable to make private member of public class package-private for testing, but it is not acceptable to raise accessibility any higher. 
 - Use package-private visibility with `@VisibleForTesting` annotation to indicate that this is package-private just because we need to test it (rare cases, this annotation is used with `public` accessor too).


__Item 16 : In public classes, use accessor methods, not public fields__

 - Public classes should never expose its fields. Doing this will prevent you to change its representation in the future. Except static final primitives or references to immutable objects (effectively constants in CAPITAL_WORDS).
 - Package private classes or private nested classes, can, on the contrary, expose their fields since it won't be part of the API.
   - For package private classes (as this is implementation detail), it can be changed or modified in code later as clients are within package only.
   - For private nested class, it doesn't really matter what the fields inside it are marked with access modifier as only outer class can read it no matter what the accessor is.

__Item 17 : Minimize mutability__

To create an immutable class : 
 - Don't provide methods that modify the visible object's state (mutators / setters)
 - Ensure that the class can't be extended (by using `final` or another way given below)
   - Can make class impossible to be subclassed by making all of its constructors private and add public static factories in place of the public constructors 
 - Make all fields `final`
 - Make all fields `private`
 - Don't give access to a reference of a mutable object that is a field of your class. Never initialize such a field to a client-provided object reference or return the field from an accessor.
 
As a rule of thumb, try to limit mutability.

 - Immutable objects are simple as it can be in exactly one state, the state in which it was created.
 - Immutable objects are inherently thread-safe; they require no synchroni￾zation, hence immutable objects can be shared freely among threads.
 - You need not and should not provide a clone method or copy constructor on an immutable class.
 - Immutable objects make great building blocks for other objects, whether mutable or immutable
 - The major disadvantage of immutable classes is that they require a separate object for each distinct value.
   - To tackle this disadvantage, the class can identify commonly used multistep operation and using an internal mutable companion class to speed them up. Or to provide a public companion class, e.g. StringBuilder for String.

```java
// instead of
String result = "";
for (int i = 0; i = 2 << 20; i++) {
  result += "hello " + i + "\n";
}

// consider, which does not produce 2M of temp strings
StringBuilder sb = new StringBuilder((2<<20)*16;
for (int i = 0; i = 2 << 20; i++) {
  sb.append("hello ").append(i).append("\n");
}
final String result = sb.toString();
```
  
 - Consider adding `Comparable` and `equals`+`hashCode` to immutable classes to simplify unit testing 
 - Immutability requires special efforts if combined with `Serializable`
 - If using non-final classes which tend to be immutable, double check that given objects are not instances of some extended class (which can be no longer immutable). E.g. BigInteger allows extensions to be mutable)
 

__Item 18 : Favor composition over inheritance__

With inheritance, you don't know how your class will react with a new version of its superclass.
For example, you may have added a new method whose signature will happen to be the same than a method of its superclass in the next release.
You will then override a method without even knowing it.

Also, if there is a flaw in the API of the superclass you will suffer from it too.
With composition, you can define your own API for your class.

As a rule of thumb, to know if you need to choose inheritance over composition, you need to ask yourself if B is really a subtype of A.

Example :
```java
// Wrapper class - uses composition in place of inheritance
public class InstrumentedSet<E> extends ForwardingSet<E> {
	private int addCount = 0;
	public InstrumentedSet (Set<E> s){
		super(s)
	}

	@Override
	public boolean add(E e){
		addCount++;
		return super.add(e);
	}

	@Override
	public boolean addAll (Collection< ? extends E> c) {
		addCount += c.size();
		return super.addAll(c);
	}

	public int getAddCount() {
		return addCount;
	}
}

// Reusable forwarding class
public class ForwardingSet<E> implements Set<E> {
	private final Set<E> s; // Composition
	public ForwardingSet(Set<E> s) { this.s = s ; }

	public void clear() {s.clear();}
	public boolean contains(Object o) { return s.contains(o);}
	public boolean isEmpty() {return s.isEmpty();}
	...
}
```

__Item 19 : Create classes for inheritance or forbid it__

First of all, you need to document all the uses of overridable methods.
Remember that you'll have to stick to what you documented.
The best way to test the design of your class is to try to write subclasses.
Never call overridable methods in your constructor.

If a class is not designed and documented for inheritance it should be me made forbidden to inherit her, either by making it final, or making its constructors private (or package private) and use static factories.


__Item 20 : Interfaces are better than abstract classes__

Since Java 8, it's possible to implements default mechanism in an interface.
Java only permits single inheritance so you probably won't be able to extends your new abstract class to exising classes when you always will be permitted to implements a new interface.

When designing interfaces, you can also provide a Skeletal implementation. This type of implementation is an abstract class that implements the interface. 
It can help developers to implement your interfaces and since default methods are not permitted to override Object methods, you can do it in your Skeletal implementation.
Doing both allows developers to use the one that will fit their needs.


__Item 21 : Design interfaces for posterity__

With Java 8, it's now possible to add new methods in interfaces without breaking old implementations thanks to default methods.
Nonetheless, it needs to be done carefully since it can still break old implementations that will fail at runtime.

__Item 22 : Interfaces are meant to define types__

Interfaces must be used to define types, not to export constants.

Example :

```java
//Constant interface antipattern. Don't do it !
public interface PhysicalConstants {
	static final double AVOGADROS_NUMBER = 6.022_140_857e23;
	static final double BOLTZMAN_CONSTANT = 1.380_648_52e-23;
	...
}
//Instead use
public class PhysicalConstants {
	private PhysicalConstants() {} //prevents instantiation
	
	public static final double AVOGADROS_NUMBER = 6.022_140_857e23;
	public static final double BOLTZMAN_CONSTANT = 1.380_648_52e-23;
	...
}
```

__Item 23 : Tagged classes__

Those kinds of classes are clutted with boilerplate code (Enum, switch, useless fields depending on the enum).
Don't use them. Create a class hierarchy that will fit you needs better.


__Item 24 : Nested classes__

If a member class does not need access to its enclosing instance then declare it static.
If the class is non static, each instance will have a reference to its enclosing instance. That can result in the enclosing instance not being garbage collected and memory leaks.

__Item 25 : One single top level class by file__

Even though it's possible to write multiple top level classes in a single file, don't !
Doing so can result in multiple definition for a single class at compile time.

## Generics

### Generic classes
```java
public class Printer<T> {
	T thingToPrint;

	public Printer(T thingToPrint) {
		this.thingToPrint = thingToPrint;
	}

	public void print() {
		System.out.println(thingToPrint);
	}
}

Printer<Integer> printer = new Printer<>(23);
printer.print();

Printer<Double> printer2 = new Printer<>(33.5);
printer2.print();

// Do note that generics don't work with primitives
Printer<int> printer3 = new Printer<>>(11); // Compiler error.
```
Bounded Generic
```java
// 1. We use `extends` with interfaces too in generics <T extends Serializable>
// 2. For multiple upper bounds use <T extends Animal & Serializable> [obviously we can extend only 1 class, but multiple interfaces]
public class Printer<T extends Animal> {
	T thingToPrint;

	public Printer(T thingToPrint) {
		this.thingToPrint = thingToPrint;
	}

	public void print() {
		System.out.println(thingToPrint);
	}
}

// Compilation error.
Printer<Integer> printer = new Printer<>(23);
printer.print();

// Works fine.
Printer<Cat> printer = new Printer<>(new Cat());
printer.print();
```

### Generic Methods
```java
public class Main {
	public static void main(String[] args) {
		shout("hello", 74);
		shout(new Cat(), "yo");
	}

	
	public static <T, V> void shout(T thingToShout, V otherThingToShout) {
		System.out.println(thingToShout + "!!");
		System.out.println(otherThingToShout + "!!");
	}
}
```
Bounded Generic
```java
public class Main {
	public static void main(String[] args) {
		shout("hello"); // Error.
		shout(new Cat());
	}
	
	public static <T extends Animal> void shout(T thingToShout) {
		System.out.println(thingToShout + "!!");
	}
}
```

Wildcard, used when type T is actually a parameter in a generic class. Because say Integer is a subclass of Object, but List<Integer> is not a subclass of List<Object>
```java
public class Main {
	public static void main(String[] args) {
		List<Cat> catList = new ArrayList<>();
		catList.add(new Cat());
		printList(catList);
	}

	// Unbounded: List<?>
	// Upper Bounded: List<? extends Animal>
	// Lower Bounded: List<? super Integer>
	public static void printList(List<? extends Animal> mylist) {
		for (Animal n : mylist) {
			System.out.println(n);
		}
	}
}
```

Generics means type safety. Rather than finding type inconsistencies during runtime, find it at compile time.

Two types of generic methods:
```java
class A<T> {
	// Because class is type parameterized, we can directly use type parameter in function declaration.
	public void foo(T value) {
	}
}

class B {
	// Class is not type parameterized, have to put <T> before return type
	public <T> void foo(T value) {
	}
}
```

```java
// This can be called as takeThing(ArrayList<Dog>), takeThing(ArrayList<Cat>), takeThing(ArrayList<Animal>) etc.
public <T extends Animal> void takeThing(ArrayList<T> list) {}

// This can only be called as takeThing(animals)
public void takeThing(ArrayList<Animal> list) {}
```

If need to sort, say a List<Song>, can do either
1. making `Song` implement `Comparable` interface and hence implementing `compareTo` method, and then just calling `Collections.sort(songList)`
2. create a comparator implementing `Comparator` which implements `compare` method, and passing this comparator by calling `Collections.sort(songList, myComparator)`

From Java Collections, we mainly get 3 things
 - `List` interface extends `Collection`
   - `ArrayList` implements `List`
   - `LinkedList` implements `List`
   - `Vector` implements `List`
 - `Set` interface extends `Collection`
   - `SortedSet` interface extends `Set`
     - `TreeSet` implements `SortedSet` (can instantiate with no-arg constructor which would mean using type parameter class's `compareTo` method (which means they must implement `Comparable`) . Or, we can also instantiate passing Comparator as arg to constructor)
   - `HashSet` implements `Set`
   - `LinkedHashSet` implements `Set`
 - `Map` interface (it **does not** extend `Collection` interface)
   - `SortedMap` interface extends `Map`
     - `TreeMap` implements `SortedMap`
   - `HashMap` implements `Map`
   - `LinkedHashMap` implements `Map`
   - `HashTable` implements `Map`
  
For `HashSet` and `HashMap`, it first calls `hashCode` method and checks if this hashcode is present in the set or not. If not, it simply means that this is a diff element altogether. If yes, it then calls `equals` method to see if object references are actually meaningfully equal or not. Hence, you should always override `hashCode` and `equals` together if overriding any of the two.
 - `a.equals(b)` must mean that `a.hashCode() == b.hashCode()`
 - `a.hashCode() == b.hashCode()` does not have to mean `a.equals(b)`

Arrays vs Generics
 - Array type checks happen at runtime
 - Generic type checks happen at compile time
 - If a method is say `foo(Animal[] animals)`, it can take all: `foo(Animal[])`, `foo(Dog[])`, `foo(Cat[])`

```java
// Array
public void foo(Animal[] animals) {
	for (Animal animal: animals) {
		animal.eat();
	}
}
 
Animal[] animals = {new Dog(), new Cat()};
foo(animals);
Dog[] dogs = {new Dog(), new Dog()};
foo(dogs);

// Generic (ArrayList)
public void foo(ArrayList<Animal> animals) {
	for (Animal animal: animals) {
		animal.eat();
	}
}

ArrayList<Animal> animals = new ArrayList<>(); animals.add(new Dog()); animals.add(new Cat());
foo(animals);
ArrayList<Dog> dogs = new ArrayList<>(); dogs.add(new Dog()); dogs.add(new Dog());
foo(dogs); // compile-error!!! 

// Java doesn't allow because of the following, say it was allowed
// call foo by passing ArrayList of Dog, and in foo add Cat to it --> YIKES!
public void foo(ArrayList<Animal> animals) {
	animals.add(new Cat());
}
foo(dogs);

// We can do above thing using arrays say reassign animal[0] = new Cat();
// but that would be then caught at runtime --> YIKES.
// Hence, preempt the errors by using Generics and not arrays

// To take a method any type of Animal using ArrayList, use either of these
public void foo(ArrayList<? extends Animal> animals) {
	// compiler won't allow any addition to `animals`
	animals.add(new Cat());

	// this is fine
	findAngriestAnimal(animals);
}

// Exactly same as above
public <T extends Animal> void foo(ArrayList<T> animals) {
	...
}
``` 
__Item 26 : Raw types__

 - A raw type is a generic type without its type parameter (Example : `List` is the raw type of `List<E>`).
 - Raw types shouldn't be used. They exist for compatibility with older versions of Java.
We want to discover mistakes as soon as possible (compile time) and using raw types will probably result in error during runtime.
```java
//Use of raw type : don't !
private final Collection stamps = ...
stamps.add(new Coin(...)); // Erroneous insertion. Does not throw any error (but gives "unchecked call" warning)
Stamp s = (Stamp) stamps.get(i); // Throws runtime ClassCastException when getting the Coin
```
 - `List<Object>` is better than `List` because if there is a `List<String>`, then it can be passed to a method which accepts `List`, but it cannot be passed to a method that accepts `List<Object>`.
 - If we don't care about type parameter in a method, then we should use unbounded wildcard types, rather than raw types. **You can't put any element (other than null) into a Collection<?>**
```java
// Bad
static int numElementsInCommon(Set s1, Set s2) {
	int result = 0;
	for (Object o : s1) {
		if (s2.contains(o)) {
			result++;
		}
	}
	return result;
}

// Good
static int numElementsInCommon(Set<?> s1, Set<?> s2) {
	...
}
``` 
We still need to use raw types in two cases : 
 - Usage of class literals (`List.class` or `String[].class`) (there is no such thing as `List<String>.class` or `List<?>.class`)
 - Usage of `instanceof`. `instanceof` does not recognize type parameters as type parameter is erased on runtime. After verifying that something is an `instanceof` `Set`, it must be casted into `Set<?>` (and not `Set`)


```java
// Common usage of instance of
if (o instanceof Set) {		// Raw type
	Set<?> = (Set<?>) o;	// Wildcard type
}
```

__Item 27 : Unchecked warnings__

Working with generics can often create warnings about them. Not having those warnings assure you that your code is typesafe.
Try as hard as possible to eliminate them. Those warnings represent a potential ClassCastException at runtime.
When you prove your code is safe but you can't remove this warning use the annotation @SuppressWarnings("unchecked") as close as possible to the declaration.
Also, comment on why it is safe.

__Item 28 : List and arrays__

Arrays are **covariant** and generics are **invariant** meaning that because `Object` is a superclass of `String`, then `Object[]` is a superclass of `String[]` when `List<Object>` is not a superclass of for `List<String>`.

```java
// Fails at runtime!
Object[] objectArray = new Long[1];
objectArray[0] = "I don't fit in"; // Throws ArrayStoreException

// Won't compile!
List<Object> ol = new ArrayList<Long>(); // Incompatible types
ol.add("I don't fit in");
```

Arrays are **reified** when generics are **erased**. Meaning that array have their typing constraint at "runtime" while generics enforce it at "compile" time and discard their type information at runtime. In order to assure retrocompatibility with previous version; `List<String>` will just be a `List` at runtime.
Typesafety is assured at compile time with generics. Since it's always better to have our coding errors the sooner (meaning at compile time), prefer the usage of generics over arrays.

__Item 29 : Generic types__ 

Generic types are safer and easier to use because they won't require any cast from the user of this type.
When creating new types, always think about generics in order to limit casts.

__Item 30 : Generic methods__

Like types, methods are safer and easier to use it they are generics. 
If you don't use generics, your code will require users of your method to cast parameters and return values which will result in non typesafe code.

__Item 31 : Bounded wildcards__

Bounded wildcards are important in order to make our code as generic as possible. 
They allow more than a simple type but also all their sons (? extends E) or parents (? super E)

Examples :

If we have a stack implementation and we want to add two methods pushAll and popAll, we should implement it this way :
```java
//We want to push in everything that is E or inherits E
public void pushAll(Iterable<? Extends E> src) {
	for (E e : src) {
		push(e);
	}
}

//We want to pop out in any Collection that can welcome E
public void popAll(Collection<? super E> dst) {
	while(!isEmpty()) {
		dst.add(pop());
	}
}
```

__Item 32 : Generics and varargs__

Even though it's not legal to declare generic arrays explicitly, it's still possible to use varargs with generics.
This inconsistency has been a choice because of its usefulness (Example : Arrays.asList(T... a)).
This can, obviously, create problems regarding type safety. 
To make a generic varargs method safe, be sure :
 - it doesn't store anything in the varargs array
 - it doesn't make the array visible to untrusted code
When those two conditions are met, use the annotation @SafeVarargs to remove warnings that you took care of and show users of your methods that it is typesafe.

__Item 33 : Typesafe heterogeneous container__

Example : 

```java
public class Favorites {
	private Map<Class<?>, Object> favorites = new HashMap<Class<?>, Object>();

	public <T> void putFavorites(Class<T> type, T instance) {
		if(type == null)
			throw new NullPointerException("Type is null");
		favorites.put(type, type.cast(instance));//runtime safety with a dynamic cast
	}

	public <T> getFavorite(Class<T> type) {
		return type.cast(favorites.get(type));
	}
}
```

## Enums and annotations

__Item 34 : Enums instead of int constants__

Prior to enums it was common to use int to represent enum types. Doing so is now obsolete and enum types must be used.
The usage of int made them difficult to debug (all you saw was int values).

Enums are classes that export one instance for each enumeration constant. They are instance controlled. They provide type safety and a way to iterate over each values.

If you need a specific behavior for each value of your enum, you can declare an abstract method that you will implement for each value.

Enums have an automatically generated valueOf(String) method that translates a constant's name into the constant. If the toString method is overriden, you should write a fromString method.

Example : 

```java
public enum Operation {
	PLUS("+") { double apply(double x, double y){return x + y;}},
	MINUS("-") { double apply(double x, double y){return x - y;}},
	TIMES("*") { double apply(double x, double y){return x * y;}},
	DIVIDE("/") { double apply(double x, double y){return x / y;}};

	private final String symbol;
	private static final Map<String, Operation>	stringToEnum = Stream.of(values()).collect(toMap(Object::toString, e -> e));
	
	Operation(String symbol) {
		this.symbol = symbol;
	}
	
	public static Optional<Operation> fromString(String symbol) {
		return Optional.ofNullable(stringToEnum.get(symbol);
	}
	
	@Override
	public String toString() {
		return symbol;
	}
	
	abstract double apply(double x, double y);
}
```

__Item 35 : Instance fields instead of ordinals__

Never use the ordinal method to calculate a value associated with an enum.

Example : 

```java
//Never do this !
public enum Ensemble {
	SOLO, DUET, TRIO, QUARTET;
	public int numberOfMusicians(){
		return ordinal() + 1;
	}
}

//Instead, do this : 
public enum Ensemble {
	SOLO(1), DUET(2), TRIO(3), QUARTET(4);
	
	private final int numberOfMusicians;
	
	Ensemble(int size) {
		this.numberOfMusicians = size;
	}
	
	public int numberOfMusicians() {
		return numberOfMusicians;
	}
}

```

__Item 36 : EnumSet instead of bit fields__

Before enums existed, it was common to use bit fields for enumerated types that would be used in sets. This would allow you to combine them but they have the same issues than int constants we saw in item 34.
Instead use EnumSet to combine multiple enums.

Example : 

```java
public class Text {
	public enum Style {BOLD, ITALIC, UNDERLINE}
	public void applyStyle(Set<Style> styles) {...}
}

//Then you would use it like this : 
text.applyStyle(EnumSet.of(Style.BOLD, Style.ITALIC));
```

__Item 37 : EnumMap instead of ordinal__

You may want to store data by a certain enum. For that you could have the idea to use the ordinal method. This is a bad practice.
Instead, prefer the use of EnumMaps.

__Item 38 : Emulate extensible enums with interfaces__

The language doesn't allow us to write extensible enums. In the few cases that we would want an enum type to be extensible, we can emulate it with an interface written for the basic enum.
Users of the api will be able to implements this interface in order to "extend" your enum.

__Item 39 : Annotations instead of naming patterns__

Prior to JUnit 4, you needed to name you tests by starting with the word "test". This is a bad practice since the compiler will never complain if, by mistake, you've names a few of them "tset*".
Annotations are a good way to avoid this kind of naming patterns and gives us more security.

Example : 

```java
//Annotation with array parameter
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface ExceptionTest {
 Class<? extends Exception>[] value();
}

//Usage of the annotation
@ExceptionTest( {IndexOutOfBoundsException.class, NullPointerException.class})
public void myMethod() { ... }

//By reflexion you can use the annotation this way 
m.isAnnotationPresent(ExceptionTest.class);
//Or get the values this way : 
Class<? extends Exception>[] excTypes = m.getAnnotation(ExceptionTest.class).value();
```

__Item 40 : Use @Override__

You should use the @Override for every method declaration that you believe to override a superclass declaration.

Example : 

```java
//Following code won't compile. Why ?
@Override
public boolean equals(Bigram b) {
	return b.first == first && b.second == second;
}

/**
This won't compile because we aren't overriding the Object.equals method. We are overloading it !
The annotation allows the compiler to warn us of this mistake. That's why @Override is really important !
**/
```

__Item 41 : Marker interfaces__

A marker interface is an interface that contains no method declaration. It only "marks" a class that implements this interface. One common example in the JDK is Serializable.
Using marker interface results in compile type checking.

## Lambdas and streams

__Item 42 : Lambdas are clearer than anonymous classes__

Lambdas are the best way to represent function objects. As a rule of thumb, lambdas needs to be short to be readable. Three lines seems to be a reasonnable limit.
Also the lambdas needs to be self-explanatory since it lacks name or documentation. Always think in terms of readability.

__Item 43 : Method references__

Most of the time, method references are shorter and then clearer. The more arguments the lambdas has, the more the method reference will be clearer.
When a lambda is too long, you can refactor it to a method (which will give a name and documentation) and use it as a method reference.

They are five kinds of method references : 

|Method ref type|Example|Lambda equivalent|
|--|--|--|
|Static|Integer::parseInt|str -> Integer.parseInt(str)|
|Bound|Instant.now()::isAfter|Instant then = Instant.now(); t->then.isAfter(t)|
|Unbound|String::toLowerCase|str -> str.toLowerCase()|
|Class Constructor|TreeMap<K,V>::new|() -> new TreeMap<K,V>|
|Array Constructor|int[]::new|len -> new int[len]|

__Item 44 : Standard functional interfaces__

java.util.Function provides a lot of functional interfaces. If one of those does the job, you should use it

Here are more common interfaces : 

|Interface|Function signature|Example|
|--|--|--|
|UnaryOperator<T>|T apply(T t)|String::toLowerCase|
|BinaryOperator<T>|T apply(T t1, T t2)|BigInteger::add|
|Predicate<T>|boolean test(T t)|Collection::isEmpty|
|Function<T,R>|R apply(T t)|Arrays::asList|
|Supplier<T>|T get()|Instant::now|
|Consumer<T>|void accept(T t)|System.out::println|

When creating your own functional interfaces, always annotate with @FunctionalInterfaces so that it won't compile unless it has exactly one abstract method.

__Item 45 : Streams__

Carefully name parameters of lambda in order to make your stream pipelines readable. Also, use helper methods for the same purpose.

Streams should mostly be used for tasks like : 
 - Transform a sequence of elements
 - Filter a sequence of elements
 - Combine sequences of elements 
 - Accumulate a sequence of elements inside a collection (perhaps grouping them)
 - Search for en element inside of a sequence

__Item 46 : Prefer side-effect-free functions in streams__
 
Programming with stream pipelines should be side effect free. 
The terminal forEach method should only be used to report the result of a computation not to perform the computation itself.
In order to use  streams properly, you need to know about collectors. The most important are toList, toSet, toMap, groupingBy and joining.

__Item 47 : Return collections instead of streams__

The collection interface is a subtype of Iterable and has a stream method. It provides both iteration and stream access.
If the collection in too big memory wise, return what seems more natural (stream or iterable)

__Item 48 : Parallelization__

Parallelizing a pipeline is unlikely to increase its performance if it comes from a Stream.iterate or the limit method is used.
As a rule of thumb, parallelization should be used on ArrayList, HashMap, HashSet, ConcurrentHashMap, arrays, int ranges and double ranges. Those structure can be divided in any desired subranged and so on, easy to work among parrallel threads.

## Methods

__Item 49 : Check parameters for validity__

When writing a public or protected method, you should begin by checking that the parameters are not enforcing the restrictions that you set.
You should also document what kind of exception you will throw if a parameter enforce those restrictions.
The *Objects.requireNonNull* method should be used for nullability checks.

__Item 50 : Defensive copies__

If a class has mutable components that comes from or goes to the client, the class needs to make defensive copies of those components.

Example : 

```java
//This example is a good example but since java 8, we would use Instant instead of Date which is immutable
public final class Period {
	private final Date start;
	private final Date end;
	/**
	* @param start the beginning of the period
	* @param end the end of the period; must not precede start;
	* @throws IllegalArgumentException if start is after end
	* @throws NullPointerException if start or end is null
	*/
	public Period(Date start, Date end) {
		this.start = new Date(start.getTime());
		this.end = new Date(end.getTime());
		if(start.compare(end) > 0) {
			throw new IllegalArgumentException(start + " after " + end );
		}
	}

	public Date start(){
		return new Date(start.getTime());
	}

	public Date end(){
		return new Date(end.getTime());
	}
	...
}
```

__Item 51 : Method signature__

Few rules to follow when designing you API :
 - Choose your methode name carefully. Be explicit and consistent.
 - Don't provide too many convenience methods. A small API is easier to learn and use.
 - Avoid long parameter lists. Use helper class if necessary.
 - Favor interfaces over classes for parameter types.
 - Prefer enum types to boolean parameters when it makes the parameter more explicit.
 
__Item 52 : Overloading__

Example : 

```java
// Broken! - What does this program print?
public class CollectionClassifier {
	public static String classify(Set<?> s) {
		return "Set";
	}
	public static String classify(List<?> lst) {
		return "List";
	}
	public static String classify(Collection<?> c) {
		return "Unknown Collection";
	}
	public static void main(String[] args) {
		Collection<?>[] collections = {
			new HashSet<String>(),
			new ArrayList<BigInteger>(),
			new HashMap<String, String>().values()
		};
	for (Collection<?> c : collections)
		System.out.println(classify(c)); // Returns "Unknown Collection" 3 times
	}
}
```

As shown in the previous example overloading can be confusing. It is recommended to never export two overloadings with the same number of parameters.
If you have to, consider giving different names to your methods. (writeInt, writeLong...)

__Item 53 : Varargs__

Varargs are great when you need to define a method with a variable number of arguments. Always precede the varargs parameter with any required parameter.

__Item 54 : Return empty collections or arrays instead of null__

Returning null when you don't have elements to return makes the use of your methods more difficult. Your client will have to check if your object is not null.
Always return an empty array or collection instead.

__Item 55 : Return of Optionals__

You should declare a method to return Optional<T> if it might not be able to return a result and clients will have to perform special processing if no result is returned.
You should never use an optional of a boxed primitive. Instead use OptionalInt, OptionalLong etc...

__Item 56 : Documentation__

Documentation should be mandatory for exported API. 

## General programming

__Item 57 : Minimize the scope of local variables__

To limit the scope of your variables, you should : 
 - declare them when first used
 - use for loops instead of while when doable
 - keep your methods small and focused
 
```java
//Idiom for iterating over a collection 
for (Element e : c) {
	//Do something with e
}

//Idiom when you need the iterator
for (Iterator<Element> i = c.iterator() ; i.hasNext() ; ) {
	Element e = i.next();
	//Do something with e
}

//Idiom when the condition of for is expensive
for (int i = 0, n = expensiveComputation() ; i < n ; i++) {
	//Do something with i
}
```

__Item 58 : For each loops instead of traditional for loops__

The default for loop must be a for each loop. It's more readable and can avoid you some mistakes.

Unfortunately, there are situations where you can't use this kind of loops : 
 - When you need to delete some elements
 - When you need to replace some elements
 - When you need to traverse multiple collections in parallel
 
__Item 59 : Use the standard libraries__

When using a standard library you take advantage of the knowledge of experts and the experience of everyone who used it before you.
Don't reinvent the wheel. Library code is probably better than code that we would write simply because this code receives more attention than what we could afford.

__Item 60 : Avoid float and double for exact answers__

Float and double types are not suited for monetary calculations. Use BigDecimal, int or long for this kind of calculation.
 
__Item 61 : Prefer primitives to boxed primitives__

Use primitives whenever you can. The use of boxed primitives is essentially for type parameters in parameterized types (example : keys and values in collections)

```java
//Can you spot the object creation ?
Long sum = 0L;
for (long i = 0 ; i < Integer.MAX_VALUE ; i++) {
	sum += i;
}
System.out.println(sum);

//sum is repeatably boxed and unboxed which cause a really slow running time.

```

__Item 62 : Avoid Strings when other types are more appropriate__

Avoid natural tendency to represent objects as Strings when there is better data types available.

__Item 63 : String concatenation__

Don't use the String concatenation operator to combine more than a few strings. Instead, use a StringBuilder.

__Item 64 : Refer to objects by their interfaces__

If an interface exists, parameters, return values, variables and fields should be declared using this interface to insure flexibility.
If there is no appropriate interface, use the least specific class that provides the functionality you need.

__Item 65 : Prefer interfaces to reflection__

Reflection is a powerful tool but has many disadvantages. 
When you need to work with classes unknown at compile time, try to only use it to instantiate object and then access them by using an interface of superclass known at compile time.

__Item 66 : Native methods__

It's really rare that you will need to use native methods to improve performances. If it's needed to access native libraries use as little native code as possible.

__Item 67 : Optimization__

Write good programs rather than fast one. Good programs localize design decisions within individual components so those individuals decisions can be changed easily if performance becomes an issue.
Good designs decisions will give you good performances.
Measure performance before and after each attempted optimization.

__Item 68 : Naming conventions__

| Identifier Type        |  Examples 								      |
|-------------------------|-----------------------------------------------|
| Package                 | org.junit.jupiter, com.google.common.collect  |
| Class or Interface      | Stream, FutureTask, LinkedHashMap, HttpServlet|
| Method or Field         | remove, groupBy, getCrc      				  |
| Constant Field          | MIN_VALUE, NEGATIVE_INFINITY      			  |
| Local Variable   		  | i, denom, houseNum          				  |
| Type Parameter 		  | T, E, K, V, X, R, U, V, T1, T2  			  |

## Exceptions

__Item 69 : Exceptions are for exceptional conditions__

Exceptions should never be used for ordinary control flow. They are designed for exceptional conditions and should be used accordingly.

__Item 70 : Checked exceptions and runtime exceptions__

Use checked exceptions for conditions from which the caller can reasonably recover.
Use runtime exceptions to indicate programming errors.
By convention, *errors* are only used by the JVM to indicate conditions that make execution impossible. 
Therefore, all the unchecked throwables you implement must be a subclass of RuntimeException.

__Item 71 : Avoid unnecessary use of checked exceptions__

When used sparingly, checked exceptions increase the reliability of programs. When overused, they make APIs painful to use.
Use checked exceptions only when you want the callers to handle the exceptional condition.
Remember that a method that throws a checked exception can't be used directly in streams.

__Item 72 : Standard exceptions__

When appropriate, use the exceptions provided by the jdk. Here's a list of the most common exceptions : 

| Exception                       |  Occasion for Use                                                              |
|---------------------------------|--------------------------------------------------------------------------------|
| IllegalArgumentException        |  Non-null parameter value is inappropriate                                     |
| IllegalStateException           |  Object state is inappropriate for method invocation                           |
| NullPointerException            |  Parameter value is null where prohibited                                      |
| IndexOutOfBoundsException       |  Index parameter value is out of range                                         |
| ConcurrentModificationException |  Concurrent modification of an object has been detected where it is prohibited |
| UnsupportedOperationException   |  Object does not support method                                                |

__Item 73 : Throw exceptions that are appropriate to the abstraction__

Higher layers should catch lower level exceptions and throw exceptions that can be explained at their level of abstraction.
While doing so, don't forget to use chaining in order to provide the underlying cause for failure.

__Item 74 : Document thrown exceptions__

Document every exceptions that can be thrown by your methods, checked or unchecked. This documentation should be done by using the @throws tag.
Nonetheless, only checked exceptions must be declared as thrown in your code.

__Item 75 : Include failure capture information in detail messages__

The detailed message of an exception should contain the values of all parameters that lead to such failure.

Example : 

```java
public IndexOutOfBoundsException(int lowerBound, int upperBound, int index) {
	super(String.format("Lower bound : %d, Upper bound : %d, Index : %d", lowerBound, upperBound, index));
	
	//Save for programmatic access
	this.lowerBound = lowerBound;
	this.upperBound = upperBound;
	this.index = index;
}

```

__Item 76 : Failure atomicity__

A failed method invocation should leave the object in the state that it was before the invocation.

__Item 77 : Don't ignore exceptions__

An empty catch block defeats the purpose of exception which is to force you to handle exceptional conditions.
When you decide with *very* good reasons to ignore an exception the catch block should contain a comment explaining those reasons and the variable should be named ignored.

## Concurrency

__Item 78 : Synchronize access to shared mutable data__

Synchronization is not guaranteed to work unless both read and write operations are synchronized.
When multiple threads share mutable data, each of them that reads or writes this data must perform synchronization.

__Item 79 : Avoid excessive synchronization__

As a rule, you should do as little work as possible inside synchronized regions.
When designing a mutable class think about whether it should be synchronized.

__Item 80 : Executors, tasks and streams__

The java.util.concurrent package added an executor framework. It contains class such as ExecutorService that can help you run Tasks in other threads.
You should refrain from using Threads and now using this framework in order to parallelize computation when needed.

__Item 81 : Prefer concurrency  utilities to wait and notify__

Using wait and notify is quite difficult. You should then use the higher level concurrency utilities such as the Executor Framework, concurrent collections and synchronizers.
 - Common concurrent collections : ConcurrentHashMap, BlockingQueue
 - Common synchronizers : CountdownLatch, Semaphore
 
__Item 82 : Document thread safety__

Every class should document its thread safety. When writing and unconditionally thread safe class, consider using a private lock object instead of synchronized methods. This will give you more flexibility.

Example : 

```java
// Private lock object idiom - thwarts denial-of-service attack
private final Object lock = new Object();

public void foo() {
	synchronized(lock) {
		...
	}
}
```

__Item 83 : Lazy initialization__

In the context of concurrency, lazy initialization is tricky. Therefore, normal initialization is preferable to lazy initialization.

On a static field you can use the lazy initialization holder class idiom :
```java
// Lazy initialization holder class idiom for static fields
private static class FieldHolder {
	static final FieldType field = computeFieldValue();
}
static FieldType getField() { return FieldHolder.field; }
```

On an instance field you can use the double-check idiom :
```java
// Double-check idiom for lazy initialization of instance fields
private volatile FieldType field;
FieldType getField() {
	FieldType result = field;
	if (result == null) { // First check (no locking)
		synchronized(this) {
			result = field;
			if (result == null) // Second check (with locking)
				field = result = computeFieldValue();
		}
	}
	return result;
}
```

__Item 84 : Don't depend on the thread scheduler__

The best way to write a robust and responsive program is to ensure that the average number of *runnable* threads is not significantly greater than the number of processors.
Thread priorities are among the least portable features of Java.

## Serialization

__Item 85 : Prefer alternatives to Java serialization__

Serialization is dangerous and should be avoided. Alternatives such as JSON should be used.
If working with serialization, try not deserialize untrusted data. If you have no other choice, use object deserialization filtering.

__Item 86 : Implement *Serializable* with great caution__

Unless a class will only be used in a protected environment where versions will never have to interoperate and servers will never be exposed to untrusted data, implementing Serializable should be decided with great care.

__Item 87 : Custom serialized form__

Use the default serialized form only if it's a reasonable description of the logical state of the object. Otherwise, write your own implementation in order to only have its logical state.

__Item 88 : Write readObject methods defensively__

When writing a readObject method, keep in mind that you are writing a public constructor and it must produce a valid instance regardless of the stream it is given.

__Item 89 : For instance control, prefer enum types to readResolve__

When you need instance control (such a Singleton) use enum types whenever possible.

__Item 90 : Serialization proxies__

The serialization proxy pattern is probably the easiest way to robustly serialize objects if those objects can't be extendable or does not contain circularities.

```java
// Serialization proxy for Period class
private static class SerializationProxy implements Serializable {
	private final Date start;
	private final Date end;

	SerializationProxy(Period p) {
		this.start = p.start;
		this.end = p.end;
	}

	private static final long serialVersionUID = 234098243823485285L; // Any number will do (Item 75)
}

// writeReplace method for the serialization proxy pattern
private Object writeReplace() {
	return new SerializationProxy(this);
}

// readObject method for the serialization proxy pattern
private void readObject(ObjectInputStream stream) throws InvalidObjectException {
	throw new InvalidObjectException("Proxy required");
}

// readResolve method for Period.SerializationProxy
private Object readResolve() {
	return new Period(start, end); // Uses public constructor
}
```
