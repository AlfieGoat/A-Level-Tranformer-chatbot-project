import React, { Component } from 'react'
import AppBar from '@material-ui/core/AppBar';
import MuiThemeProvider from '@material-ui/core/styles/MuiThemeProvider';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import { Toolbar, Typography, Grid, Fab } from '@material-ui/core';
import Box from '@material-ui/core/Box';
import EditIcon from '@material-ui/icons/Edit';


export class Chatbot extends Component {
    user = []
    bot = []
    text_field_value = ""
    state = {text_field: "",}

    send_message = () => {  // Used to send messages
        var ws = new WebSocket('ws://localhost:8765')  // Creates websocket 
        ws.onopen = () => {
            ws.send('{"user":' + JSON.stringify(this.user) + ', "bot":' + JSON.stringify(this.bot) + '}')
        }
        ws.onmessage = e => {  // When message is received, it writes it
            this.bot[this.bot.length-1] = e.data
            this.setState({text_field: ""})  // Forces react to re render
            ws.close()
        }
    }

    on_click = () => {  // Handles the click 
        this.user.push(this.state.text_field)  // Pushes the user's msg to the user array
        this.bot.push("thinking...")  // Writes thinking for the bot, whilst the bot is generating a response
        this.setState({text_field: ""})  // Sets the text field to an empty string
        this.send_message()  // Calls the send message command
    }

    on_change = (e) => {
        this.setState({text_field: e.target.value})  // Updates the state to match what is in the text field
    }

    reset = () => {
        this.user = []
        this.bot = []
        this.setState({text_field: ""})  // Forces react to re render

    }

    render() {
        return (
            <MuiThemeProvider>
                <React.Fragment>
                    <AppBar position="static" title="Chatbot">
                        <Toolbar>
                            <Typography variant="h4" >Chatbot</Typography>
                            <Grid container justify="flex-end">
                                <Button variant="contained" onClick={this.reset}>Reset Chat</Button>
                            </Grid>
                        </Toolbar>
                    </AppBar>
                    <br/>
                    <Typography variant="h5" color="primary">You: '{this.user[this.user.length-4]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="secondary">Bot: '{this.bot[this.bot.length-4]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="primary">You: '{this.user[this.user.length-3]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="secondary">Bot: '{this.bot[this.bot.length-3]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="primary">You: '{this.user[this.user.length-2]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="secondary">Bot: '{this.bot[this.bot.length-2]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="primary">You: '{this.user[this.user.length-1]}'</Typography>
                    <br/>
                    <Typography variant="h5" color="secondary">Bot: '{this.bot[this.bot.length-1]}'</Typography>
                    <br/>
                    <br/>
                    <Box mx={22} justify="center">
                    <TextField 
                        style={{maxHeight: '60px', minHeight: '60px', minWidth: '50%'}}
                        label="Enter msg:"
                        variant="filled"
                        onChange={this.on_change} 
                        value={this.state.text_field}
                    />
                    <Button height={512} variant="outlined" style={{maxHeight: '56px', minHeight: '56px'}} onClick={this.on_click}>Send</Button>

                    </Box>
                </React.Fragment>
            </MuiThemeProvider>
        )
    }
}

export default Chatbot
