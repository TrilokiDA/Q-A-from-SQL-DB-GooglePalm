# Created by trilo at 26-01-2024
import streamlit as st
from main import main
from flask import Flask, request

new_chain = main()

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    query = request.get_json()['query']
    otput = new_chain(query)
    return otput

if __name__ == '__main__':
    app.run(debug=True)