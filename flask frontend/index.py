from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import requests
import json

app = Flask(__name__)
chabot_convo = []

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/chatbot_interact', methods=['POST', 'GET'])
def chatbot_interact():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            chabot_convo.append('YOU: ' + sent)
            reply = 'static reply'
            chabot_convo.append('BOT: ' + reply)
    return render_template('chatbot_interact.html', conversations=chabot_convo)



'''
@app.route('/chatbot_reply', methods=['POST', 'GET'])
def chatbot_reply():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'level' not in request.json or 'dialogs' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        level = request.json['level']
        dialogs = request.json['dialogs']
    else:
        sentence = request.args.get('sentence')
        level = request.args.get('level')
        dialogs = request.args.get('dialogs')

    target_text = sentence
    target_text = 'static response'
    return jsonify({
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    })
'''


# run Flask app
if __name__ == "__main__":
    app.run(debug = True)