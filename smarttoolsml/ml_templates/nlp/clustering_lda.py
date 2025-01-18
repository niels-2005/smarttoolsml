from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def lda_example(df):
    # Vectorizing X
    count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
    X = count.fit_transform(df["review"].values)

    # LDA Clustering with n_components = 10
    lda = LatentDirichletAllocation(
        n_components=10, random_state=123, learning_method="batch", n_jobs=-1
    )
    X_topics = lda.fit_transform(X)

    # what the code below does?
    # Topic 1:
    # effects horror budget worst low special awful terrible stupid script
    # Topic 2:
    # minutes guy money watched dvd ll maybe let wasn worst
    # Topic 3:
    # war american history country documentary men german political black america
    # Topic 4:
    # human art feel cinema audience game different sense music beautiful
    # Topic 5:
    # series episode tv school episodes comedy shows family season high
    # Topic 6:
    # horror woman house girl sex killer wife women goes dead
    # Topic 7:
    # role performance comedy plays played music actor musical performances wonderful
    # Topic 8:
    # action john western hero michael star battle king fight town
    # Topic 9:
    # book script version original read production role actor novel poor
    # Topic 10:
    # action kids children fun loved animation enjoy disney family recommend
    n_top_words = 10
    feature_names = count.get_feature_names_out()

    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx + 1))
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
            )
        )

    # view some reviews to test clustering algorithm
    horror = X_topics[:, 5].argsort()[::-1]
    for iter_idx, movie_idx in enumerate(horror[:3]):
        print("\nHorror movie #%d:" % (iter_idx + 1))
        print(df["review"][movie_idx][:300], "...")
