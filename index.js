const express = require("express");
const ejs = require("ejs");
const path = require('path');
const bodyParser = require('body-parser');
const app = express();

app.use(express.json());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.set("view engine", "ejs");

app.use(express.static(path.join(__dirname, "/public")));

const PORT = process.env.PORT || 3000;

app.get("/", (req, res) => {
    res.render("index");
});

app.get("/search", (req, res) => {
    const query = req.query;
    const question = query.question;
    console.log(question);

    //TF-IDF Algo
    setTimeout(() => {
        let spawn = require('child_process').spawn;
        let process = spawn('python', ['index.py', question]);
        process.stdout.on('data', function (data) {
            str = data.toString();
            res.json(JSON.parse(str));
        });
    }, 1000);
});

app.listen(PORT, () => {
    console.log("Server is running on port" + PORT);
});