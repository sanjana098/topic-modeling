from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora


def clean_doc(doc):
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()
	cleaned_doc = " ".join([word for word in doc.lower().split() if word not in stop])
	cleaned_doc = "".join(ch for ch in cleaned_doc if ch not in exclude)
	cleaned_doc = " ".join(lemma.lemmatize(word) for word in cleaned_doc.split())

	return cleaned_doc

d1 = "If we all want to make AI-driven products that solve real problems and are sustainable businesses, we need the best. This is going to require a variety of minds on projects, and that means increasing the number of women on engineering teams."
d2 = "I however will argue here about something beyond the need for diversity. I will argue that our A.I. future should be led my women and not by men. The reason for this is that women have a greater intuitive understanding of what makes us all human. Women have a natural inclination to focus on the important things that make us human. To maximize the benefit of AI technology we must focus on how AI improves our humanity and therefore we need to understand, at the very least, what makes us human and not what makes us machines."
d3 = "On a local scale, AI will offer opportunities to make our lives safer, more convenient, and more satisfying. That means automated cars that can drive us to and from work, or prevent life-threatening accidents when our teenagers are at the wheel. It means customized healthcare, built using knowledge gleaned from enormous amounts of data. And counter to common knowledge, it means more satisfying jobs, not less, as the productivity gains from AI and robotics free us up from monotonous tasks and let us focus on the creative, social, and high-end tasks that computers are incapable of."
d4 = "On a global scale, AI will help us generate better insights into addressing some of our biggest challenges: understanding climate change by collecting and analyzing data from vast wireless sensor networks that monitor the oceans, the greenhouse climate, and the plant condition; improving governance by data-driven decision making; eliminating hunger by monitoring, matching and re-routing supply and demand, and predicting and responding to natural disasters using cyber-physical sensors. It will help us democratize education through MOOC offerings that are adaptive to student progress, and ensure that every child gets access to the skills needed to get a good job and build a great life."
d5 = "Artificial-Intelligence (AI) is the new electricity."

docs = [ d1, d2, d3, d4, d5	]

cleaned_doc = [clean_doc(doc).split() for doc in docs]

dictionary = corpora.Dictionary(cleaned_doc)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_doc]

Lda = gensim.models.ldamodel.LdaModel 
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word=dictionary, passes=70)

print ldamodel.print_topics(num_topics=3, num_words=1)