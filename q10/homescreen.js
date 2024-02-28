import React from 'react';
import { View, Text, Button } from 'react-native';
import * as firebase from 'firebase';

const HomeScreen = ({ navigation }) => {
  const handleSignOut = () => {
    firebase.auth().signOut();
  };

  return (
    <View>
      <Text>Welcome to the Volunteer App!</Text>
      <Button title="Sign Out" onPress={handleSignOut} />
    </View>
  );
};

export default HomeScreen;
