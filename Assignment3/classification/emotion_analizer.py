import spacy

nlp = spacy.load("en_core_web_sm")

emotional_words = "anger sad happy joy"
emotional_tokens = list(nlp.pipe(emotional_words))

def emotionality_component(doc):
    score = 0.
    count = 0
    for token in doc:
        max = 0
        if(token.is_stop): continue
        count += 1
        for emotional_token in emotional_tokens:
            sim = token.similarity(emotional_token)
            if(sim > max): max = sim
        score += max
    doc.user_data['emotionality'] = score/count
    return doc


def emotionality_component_doc(doc):
    score = 0.
    for emotional_token in emotional_tokens:
        sim = doc.similarity(emotional_token)
        if(sim > score): score = sim
    doc.user_data['emotionality_doc'] = score/len(doc)
    return doc

nlp.add_pipe(emotionality_component, last=True)
#nlp.add_pipe(emotionality_component_doc, last=True)

def findBestEmotional(texts: [str]):
    scores = []

    docs = nlp.pipe(texts, batch_size=100, n_threads=8)
    #docs = list(reversed(sorted(docs, key=lambda d: d.user_data['emotionality'], reverse=True)))
    docs = list(sorted(docs, key=lambda d: d.user_data['emotionality'], reverse=True))
    #docs2 = list(sorted(docs, key=lambda d: d.user_data['emotionality_doc'], reverse=True))

    return docs
    #for text in texts:
    #    scores.append((emotionality(text), text.emotionality))

    #return list(reversed(sorted(scores, key=lambda tup: tup[0])))

def emotionality(text):
    tokens = nlp(text)
    score = 0
    for token in tokens:
        for emotional_token in emotional_tokens:
            score += token.similarity(emotional_token)

    return score/len(tokens)


