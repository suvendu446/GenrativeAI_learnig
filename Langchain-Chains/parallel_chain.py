import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model_1= ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

model_2 = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)



prompt1 = PromptTemplate(
    template = "generate a short and simple notes from the following text \n {text}",
    input_variables = ["text"], 
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question from the following text \n {text}",
    input_variables = ["text"], 
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and questions into a single text \n notes: {notes} \n questions: {questions}",
    input_variables = ["notes", "questions"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model_1 | parser,
    'questions' : prompt2 | model_2 | parser
})

merge_chain = prompt3 | model_1 | parser

chain = parallel_chain | merge_chain

text = "Machine Learning Knowledge Base: Core Concepts and ArchitecturesModule 1: Introduction to Machine Learning Paradigms1.1 Supervised LearningSupervised learning is a machine learning paradigm where the model is trained on labeled data. The dataset consists of input-output pairs, denoted as \((x_i, y_i)\), where \(x_{i}\) represents the feature vector and \(y_{y}\) represents the ground-truth label. The primary objective is to learn a mapping function f: X → Y such that the predicted output ŷ = f(x) minimizes a predefined loss function. Supervised learning tasks are broadly categorized into two types: classification and regression. Classification deals with discrete target variables, such as predicting whether an email is spam or ham. Regression deals with continuous target variables, such as predicting housing prices based on geographical and structural features. Common algorithms include Linear Regression, Support Vector Machines (SVM), Decision Trees, and Random Forests.1.2 Unsupervised LearningUnsupervised learning involves training models on datasets without explicit labels. The system attempts to discover underlying patterns, structures, or anomalies directly from the input features \(x_{i}\). Unlike supervised learning, there is no error signal or feedback loop to evaluate the absolute correctness of the output. The most prominent task in unsupervised learning is clustering, which groups similar data points together based on distance metrics like Euclidean or Manhattan distance. K-Means clustering, Hierarchical Clustering, and DBSCAN are widely used algorithms. Another critical subfield is dimensionality reduction, which projects high-dimensional data into a lower-dimensional space while preserving essential variance. Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) are classic examples used to fight the curse of dimensionality.1.3 Reinforcement LearningReinforcement Learning (RL) operates on a behavioral framework where an autonomous agent learns to make decisions by interacting with an environment. The framework is formally modeled as a Markov Decision Process (MDP), defined by a tuple \((S, A, P, R, \gamma)\), representing states, actions, transition probabilities, rewards, and a discount factor. The agent receives a reward or penalty based on the action it executes, intending to maximize the cumulative expected future reward over time. Through a process of exploration (trying unknown actions) and exploitation (using known successful actions), the agent optimizes its policy function π(a|s). Key architectures include Q-Learning, Deep Q-Networks (DQN), and Policy Gradient methods.Module 2: Deep Learning Essentials and Neural Networks2.1 Foundations of Artificial Neural NetworksArtificial Neural Networks (ANNs) are computational models inspired by biological neural networks. The fundamental building block is the perceptron, which takes multiple weighted inputs, adds a bias term, and passes the aggregate sum through a non-linear activation function. Without non-linear activation functions like ReLU (Rectified Linear Unit), Sigmoid, or Tanh, a multi-layer neural network collapses into a simple linear transformation, rendering it incapable of learning complex decision boundaries. Deep Neural Networks (DNNs) consist of an input layer, multiple hidden layers, and an output layer. Optimization is achieved via backpropagation, which uses the chain rule of calculus to compute the gradient of the loss function with respect to each weight in the network. Gradient Descent or variants like Adam and RMSprop then update these weights to minimize overall error.2.2 Convolutional Neural Networks (CNNs)Convolutional Neural Networks are specialized deep learning architectures designed primarily for processing structured grid data, such as images. Standard fully connected networks fail on image data due to the massive parameter explosion and a lack of spatial invariance. CNNs solve this by utilizing convolutional layers, pooling layers, and fully connected layers. The convolutional layer applies learnable filters (kernels) that slide across the input space to create feature maps, capturing local spatial patterns like edges, textures, and shapes. Pooling layers (such as Max Pooling) reduce the spatial dimensions of the feature maps, downsampling the data to achieve translation invariance and decrease computational overhead. CNNs power modern computer vision tasks, including facial recognition, object detection, and autonomous driving systems.2.3 Recurrent Neural Networks (RNNs) and Sequential DataRecurrent Neural Networks are tailored for sequential data streams where past context influences current predictions, such as natural language processing, time-series forecasting, and speech recognition. Unlike feedforward networks, RNNs maintain a hidden state vector that acts as a memory buffer across sequential time steps. However, standard RNNs suffer from vanishing and exploding gradient problems when trained on long sequences, making them unable to retain long-term dependencies. To mitigate this, advanced gated architectures were introduced. Long Short-Term Memory (LSTM) networks feature an internal cell state regulated by three gates: an input gate, a forget gate, and an output gate. Gated Recurrent Units (GRUs) offer a streamlined alternative by combining the cell state and hidden state, using fewer parameters while maintaining comparable performance.Module 3: MLOps and Lifecycle Management3.1 Data Pipeline EngineeringThe performance of any machine learning system is bounded by the quality of its data infrastructure. Data pipeline engineering involves the collection, ingestion, cleaning, transformation, and storage of raw datasets. Feature engineering is a critical phase where raw variables are converted into highly predictive features. This includes handling missing values via imputation, encoding categorical variables through one-hot encoding or target encoding, and scaling numerical variables using normalization or standardization. Additionally, modern production systems utilize Feature Stores to serve as a centralized repository for storing, documenting, and sharing features across training and inference pipelines, ensuring data consistency and preventing training-serving skew.3.2 Model Deployment StrategiesDeploying machine learning models to production requires robust software engineering practices to ensure low latency, high availability, and scalability. Models are typically serialized into formats like ONNX, TensorFlow SavedModel, or Pickle, and containerized using tools like Docker. Common deployment strategies include REST API microservices, gRPC frameworks, or serverless functions. For high-volume streaming data, models are integrated directly with message brokers like Apache Kafka to perform real-time stream inference. In contrast, batch inference systems process large quantities of accumulated data offline at scheduled intervals. Organizations often use canary deployments or blue-green deployment strategies to roll out new models gradually, mitigating the risk of system-wide failures.3.3 Continuous Monitoring and Drift DetectionOnce a machine learning model is operational in production, its predictive accuracy inevitably degrades over time due to changing real-world dynamics. Continuous monitoring systems track performance metrics alongside structural data shifts. Data drift occurs when the statistical properties of the input features change over time (e.g., shifts in consumer behavior). Concept drift occurs when the statistical relationship between the input features and the target variable itself changes (e.g., a fraud detection model failing because fraudsters invented entirely new techniques). Statistical tests such as the Kolmogorov-Smirnov test or Population Stability Index (PSI) are deployed to detect these anomalies, automatically triggering automated retraining loops when drift thresholds are breached."

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()






