// App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import axios from 'axios';

import Login from './components/Login';
import Register from './components/Register';
import TaskList from './components/TaskList';

function App() {
    const [loggedIn, setLoggedIn] = useState(false);

    useEffect(() => {
        // Check if user is logged in
        const checkLoggedIn = async () => {
            try {
                const response = await axios.get('/api/auth/checkLoggedIn');
                if (response.data.loggedIn) {
                    setLoggedIn(true);
                }
            } catch (error) {
                console.error('Error checking login status:', error);
            }
        };

        checkLoggedIn();
    }, []);

    return (
        <Router>
            <Switch>
                <Route exact path="/">
                    {loggedIn ? <TaskList /> : <Login />}
                </Route>
                <Route path="/register" component={Register} />
            </Switch>
        </Router>
    );
}

export default App;
