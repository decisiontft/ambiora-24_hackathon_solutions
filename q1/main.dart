import 'dart:math';

class User {
  final String username;
  final String password;

  User(this.username, this.password);
}

class Task {
  final int taskId;
  final String taskDescription;

  Task(this.taskId, this.taskDescription);
}

class ScavengerHuntApp {
  List<User> users = [
    User('user1', 'password1'),
    User('user2', 'password2'),
    User('user3', 'password3'),
  ];

  List<Task> tasks = [
    Task(1, 'Take a selfie with a street sign.'),
    Task(2, 'Find a red flower and take a picture.'),
    Task(3, 'Write a short poem about nature.'),
    Task(4, 'Find a unique-shaped cloud and describe it.'),
    // Add more tasks as needed
  ];

  Map<User, List<Task>> userTasks = {};

  void login(String username, String password) {
    User currentUser = users.firstWhere((user) => user.username == username && user.password == password, orElse: () => null);
    if (currentUser != null) {
      if (!userTasks.containsKey(currentUser)) {
        userTasks[currentUser] = [];
        assignRandomTasks(currentUser);
      }
      print('Login successful! Welcome, ${currentUser.username}');
    } else {
      print('Invalid username or password');
    }
  }

  void assignRandomTasks(User user) {
    List<Task> availableTasks = List.from(tasks);
    availableTasks.shuffle();
    userTasks[user] = availableTasks.take(3).toList(); // Assign 3 random tasks to the user
  }

  void submitTask(User user, Task task, {String submission}) {
    if (userTasks[user].contains(task)) {
      print('Task submitted by ${user.username}: ${task.taskDescription}');
      if (submission != null) {
        print('Submission: $submission');
      }
      // Save submission to database
    } else {
      print('Task not assigned to ${user.username}');
    }
  }
}

void main() {
  ScavengerHuntApp app = ScavengerHuntApp();
  app.login('user1', 'password1');
  app.submitTask(app.users[0], app.tasks[0], submission: 'Selfie with street sign');
}
