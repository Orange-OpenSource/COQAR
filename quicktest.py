from server import server

data = [
    [
        "What is the name of Harry Potter's author?",
        "JK Rowling.",
        "When did she die?",
        "I don't know",
        "What was her favorite color?"
    ],
    [
        "What is the name of Harry Potter's author?",
        "JK Rowling.",
        "When did she die?",
        "I don't know",
        "What was her last meal?"
    ],
    [
        "What is the name of Harry Potter's author?",
        "JK Rowling.",
        "When did she die?",
        "I don't know",
        "Who wanted her dead?"
    ],
    ["Hello, what's your name?"],
    ["Hello, what's my name?"],
    ["Hello, what's the name of Emmanuel Macron?"],
    ["Hello, what is a name?"],
]

data = [
    {
        'context' : x[:-1],
        'sentence' : x[-1]
    }
    for x in data
]

s = server.Server()
s.load_model('data/trained_models/best_mixed_model')
s.rewrite_batch(instances=data)
print(s)
