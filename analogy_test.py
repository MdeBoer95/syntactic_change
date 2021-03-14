import sys,numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# python analogy_test.py /storage/nllg/compute-share/deboer/melvin/language_change/Language-change/german/embedding_change/1600-1700/word2vec/1617_emb_cleaned_mapped.txt 

def read(fn,header=True,normalize=True):
  h,hrev = {},{}
  m = []
  for iline,line in enumerate(open(fn)):
    if iline==0 and header: continue
    x = line.split()
    word = x[0]
    vec = np.array(x[1:],dtype=float)
    if normalize: vec /= np.linalg.norm(vec)
    h[word] = iline-1
    hrev[iline-1] = word
    m.append(vec)
  return h,hrev,m

def nn(m1,m2,index,words,topn=10):
  vec = np.dot(m1[index],np.transpose(m2))
  #print(words,vec.shape)
  x = [(vec[i],words[i]) for i in range(vec.shape[0])]
  x = sorted(x,reverse=True)
  return x[:topn]

def getFrequent(fn):
  lst = []
  for line in open(fn):
    line = line.strip()
    w = line.split()[0]
    lst.append(w)
  return lst

if __name__ == "__main__":

  from sklearn.metrics.pairwise import cosine_similarity as cossim

  h1,h1rev,m1 = read(sys.argv[1])
  #h2,h2rev,m2 = read(sys.argv[2])

  test_cases = [
    # verb infinitiv
    ("lachen","sagen","weinen",["kennen","kannte","kennst","kenne"]),
    ("wagen","wissen","mahlen",["bewahren","bewahrte","bewahre","bewahrt"]),
    # genetiv
    ("Mannes","Kindes","Wassers",["Hauses","Hause","Haus","Häuser"]),
    ("Mannes","Kindes","Wassers",["Lichtes","Lichte","Licht","Lichter"]),
    # dativ
    ("Walde","Kinde","Gotte",["Hauses","Hause","Haus","Häuser"]),
    ("Wege","Schwerte","Manne",["Lichtes","Lichte","Licht","Lichter"]),
    # plural
    ("Frauen","Männer","Tiere",["Götter","Gotte","Gott","Gottes"]),
    ("Augen","Häuser","Bilder",["Männer","Manne","Mann","Mannes"]),
    # akkusativ
    ("Manne","Kinde","Lichte",["Hause","Hauses","Haus","Häuser"]),
    # past verbs
    ("spielte","rannte","wusste",["sagte","sagen","sage","sagst"]),
    ("trug","schwang","machte",["redete","reden","rede","redest"]),
    ("schlug","schlang","brachte",["wollte","wolle","wollen","willst"]),
    ("konnte","ging","glaubte",["behauptete","behaupten","behaupte","behauptest"]),
    # partizip
    ("gehend","lachend","fragend",["singend","singen","sang","singst"]),
    ("machend","wissend","bestätigend",["glaubend","glauben","glaubte","glaubst"]),
    # konjunktiv / 1st person verbs
    ("spiele","renne","wisse",["sage","sagen","sagte","sagst"]),
    ("trage","frage","schwinge",["bringe","bringen","brachte","bringst"]),
    ("kennest","glaubest","sagest",["wissest","wissen","wusste","weißt"]),
    # adjektives nominative
    ("schönes","reiches","kleines",["großes","große","groß","großer"]),
    # adjective superlative
    ("größte","weiteste","niedrigste",["schönste","schön","schöner","schönes"]),
    ("ältester","jüngster","kleinster",["neuester","neu","neues","neuer"]),
    ("schnellsten","dümmsten","weißesten",["reichsten","reich","reichster","reiches","reichen"])
]

  satisfied,total = 0,0
  vals = []

  for analogy in test_cases:
    w1,w2,w3,alternatives = analogy
    ww1 = h1.get(w1,None)
    ww2 = h1.get(w2,None)
    ww3 = h1.get(w3,None)
    if ww1 is None or ww2 is None or ww3 is None: continue
    v1,v2,v3 = m1[ww1],m1[ww2],m1[ww3]

#    my_as = []
#    vec = v1-v2+v3
#    scores = []
#    for a in alternatives:
#      aa = m1[h1[a]]
#      my_as.append((np.dot(aa,vec),a))
#      scores.append(np.dot(aa,vec))
#    print(analogy,my_as)
#    total += 1
#    if scores[0]==max(scores): satisfied+=1
#    vals.append(scores[0]-max(scores[1:]))

    for (v,w) in [(v1,w1),(v2,w2),(v3,w3)]:
      scores = []
      for a in alternatives:
        wa = h1.get(a,None)
        if not wa:
            exit(1)
        aa = m1[wa]
        sim = cosine_similarity(aa.reshape(1, 100),v.reshape(1, 100))
        scores.append(sim)
        print(w,a,"%.03f"%sim)
      if scores[0]==max(scores): 
        satisfied+=1
        print("***")
      else:
        with open('analogy_test.txt', 'a') as tf:
          tf.write("Did not pass: word: " + str(w) + "Alternatives:" + str(alternatives) + '\n')
      vals.append(scores[0]-max(scores[1:]))
      total += 1
 
  print("Passed: {}/{}; Distance: {}".format(satisfied,total,np.mean(vals)))
  
  
    #vec = v1-v2+v3
    
    

  #word = sys.argv[3]
  #words = getFrequent(sys.argv[3])

  #sims = np.dot(m1,np.transpose(m1))
  #print(sims.shape)
  #print(nn(m1,m2,h1[word],h1rev))

  #for word in words:
  #  for v in words:
  #    v1 = m1[h1[word]]
  #    v2 = m2[h2[v]]
  #    print(word,v,np.dot(v1,v2))

  #print(cossim(v1,v2))
