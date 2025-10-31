import streamlit as st
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.varLib.instancer import verticalMetricsKeptInSync
import helper  # Import only helper functions, no variables!

import preprocessor  # Custom module to preprocess WhatsApp chat data

import streamlit as st

st.sidebar.markdown(
    """
    <div style="
        text-align: center; 
        font-size: 40px; 
        font-weight: bold; 
        padding: 20px;
        font-color:#ffe5d9;
    ">
        ChatIQ
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("Analyze Your WhatsApp Chat History. Follow These Steps:")
st.sidebar.markdown("1. Export The Chat File From WhatsApp.\n"
                    "2. Upload The `.txt` File Using The Uploader Below.\n"
                    "3. Click On Show Analysis."
                    )
uploaded_file = st.sidebar.file_uploader("Upload Your Chat File (e.g., .txt)", type=['txt'])

if uploaded_file is not None:
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    # Convert to string
    data = bytes_data.decode("utf-8")
    # Preprocess data
    df = preprocessor.preprocess(data)
    #st.dataframe(df)

    # Inside app.py (after df is created and preprocessed)

    # Inside app.py (after df = preprocessor.preprocess(data))

    st.sidebar.header("🔍 Message Filters")

    # 1. Keyword search
    keyword = st.sidebar.text_input("Search keyword")

    # 2. User selection
    users = df['User'].unique().tolist()
    users.sort()
    users.insert(0, "Overall")  # option for all users
    selected_users = st.sidebar.multiselect("Select user(s)", users, default=["Overall"])

    # --- Apply Filters ---
    filtered_df = df.copy()

    # User filter
    if "Overall" not in selected_users:
        filtered_df = filtered_df[filtered_df["User"].isin(selected_users)]

    # Keyword filter
    if keyword:
        filtered_df = filtered_df[filtered_df["Message"].str.contains(keyword, case=False, na=False)]

    # --- Show Results ---
    st.subheader("📋 Filtered Messages")
    st.write(f"Total messages found: {filtered_df.shape[0]}")
    st.dataframe(filtered_df[["date", "User", "Message"]].reset_index(drop=True))


    # Extract User Names
    user_list = df['User'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_User = st.sidebar.selectbox("Show Analysis for Different User", user_list)

    # Display Statistics
    if st.sidebar.button("Show Analysis"):
        num_messages, words, media_count, links, total_chars, avg_message_length = helper.fetch_stats(selected_User, df)
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">🔢 Overall Statistics</h1>', unsafe_allow_html=True)

        # Display the metrics side by side
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Total Messages</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{num_messages}</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Total Words</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{words}</div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Media Shared</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{media_count}</div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Links Shared</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{links}</div>
            """, unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Total Characters</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{total_chars}</div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
                <div style="text-align:center; font-size:22px; font-weight:bold; color:#a7c957;">Avg Msg Length</div>
                <div style="text-align:center; font-size:50px; font-weight:bold; color:#faedcd;">{avg_message_length:.2f}</div>
            """, unsafe_allow_html=True)

        # Monthly Timeline
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">📅 Monthly Timeline</h1>', unsafe_allow_html=True)

        timeline = helper.monthly_timeline(selected_User, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['Message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">🗓️ Daily Timeline</h1>', unsafe_allow_html=True)

        daily_timeline = helper.daily_timeline(selected_User, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['Message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Activity map
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">🗺️ Activity Map</h1>', unsafe_allow_html=True)

        col1,col2=st.columns(2)
        with col1:
            st.header("Most Busy Days")
            busy_day=helper.weak_activity_map(selected_User,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Months")
            busy_month=helper.month_activity_map(selected_User,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">📡 Weekly Activity Map</h1>', unsafe_allow_html=True)

        user_heatmap=helper.activity_heatmap(selected_User,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)


        # Most Busy User Analysis
        if selected_User == "Overall":
            st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">🔥 Most Busy Users</h1>', unsafe_allow_html=True)

            x, new_df = helper.most_busy_user(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # ------------------- Average Response Time -------------------
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">⏱️ Average Response Time Per User</h1>',
                    unsafe_allow_html=True)

        # Call helper function
        avg_resp = helper.avg_response_time(df, selected_User)

        if not avg_resp.empty:
            # 5 fastest responders
            fastest_5 = avg_resp.nsmallest(5)
            st.markdown('<h3 style="color: #a7c957;">Fastest 5 Responders</h3>', unsafe_allow_html=True)
            st.dataframe(
                fastest_5.reset_index().rename(columns={'User': 'User', 'delta_min': 'Avg Response Time (min)'}))

            fig, ax = plt.subplots()
            ax.barh(fastest_5.index, fastest_5.values, color='green')
            ax.set_xlabel("Avg Response Time (minutes)")
            ax.set_ylabel("User")
            ax.set_title("5 Fastest Responders")
            st.pyplot(fig)

            # 5 slowest responders
            slowest_5 = avg_resp.nlargest(5)
            st.markdown('<h3 style="color: #ff6f61;">Slowest 5 Responders</h3>', unsafe_allow_html=True)
            st.dataframe(
                slowest_5.reset_index().rename(columns={'User': 'User', 'delta_min': 'Avg Response Time (min)'}))

            fig2, ax2 = plt.subplots()
            ax2.barh(slowest_5.index, slowest_5.values, color='red')
            ax2.set_xlabel("Avg Response Time (minutes)")
            ax2.set_ylabel("User")
            ax2.set_title("5 Slowest Responders")
            st.pyplot(fig2)

        else:
            st.info("No response time data available for this selection.")

        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">💬 Sentiment Analysis</h1>', unsafe_allow_html=True)

        # Call helper function
        df_sentiment, avg_sentiment = helper.add_sentiment(df, selected_User)

        # Message-level sentiment distribution
        st.markdown("### Sentiment Distribution (All Messages)")
        fig, ax = plt.subplots()
        ax.hist(df_sentiment['sentiment'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Number of Messages")
        st.pyplot(fig)

        # Per-user average sentiment (only if overall selected)
        if avg_sentiment is not None:
            st.markdown("### Average Sentiment Per User")
            fig2, ax2 = plt.subplots()
            ax2.barh(avg_sentiment.index, avg_sentiment.values, color='orange')
            ax2.set_xlabel("Average Sentiment Score")
            ax2.set_ylabel("User")
            st.pyplot(fig2)


        # Word Cloud
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">☁️ Word Cloud</h1>', unsafe_allow_html=True)

        df_wc = helper.create_wordcloud(selected_User, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #find most common words
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">📝 Most Common Words</h1>', unsafe_allow_html=True)
        most_common_df=helper.most_common_words(selected_User,df)
        fig,ax=plt.subplots()
        ax.barh(most_common_df['Word'],most_common_df['Count'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # 😊 Emoji Analysis
        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">😊 Emoji Analysis</h1>', unsafe_allow_html=True)

        emoji_df = helper.emoji_helper(selected_User, df)
        emoji_df.columns = ['Emoji', 'Count']

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            import plotly.express as px

            top_10_emojis = emoji_df.head(10)

            # Create pie chart
            fig = px.pie(
                top_10_emojis,
                names="Emoji",
                values="Count",
                hole=0  # filled center (no donut)
            )

            # Update traces
            fig.update_traces(
                textinfo="percent+label",
                textfont_size=14,
                hoverinfo='label+percent',
                marker=dict(line=dict(color='#000000', width=2))
            )

            # Add proper title and styling
            fig.update_layout(
                title={
                    'text': "Top 10 Most Used Emojis 😊",
                    'y': 0.97,
                    'x': 0.5,  # Center align title
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                title_font=dict(size=22, color="white"),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=60, b=0, l=0, r=0)
            )

            st.plotly_chart(fig, use_container_width=True)


        # --- TOPIC MODELING / CHAT THEMES ---
        import streamlit as st
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        import pandas as pd
        import re

        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">🎯 Chat Themes / Topic Modeling</h1>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Clean messages (exclude system and empty ones)
            clean_df = df.copy()
            clean_df = clean_df[clean_df["User"] != "group_notification"]
            clean_df = clean_df[
                ~clean_df["Message"].str.contains(r"Media omitted|end-to-end encrypted|added|removed", case=False,
                                                  na=False)]
            clean_df = clean_df[clean_df["Message"].str.strip() != ""]

            if not clean_df.empty:
                # Combine all text
                messages = clean_df["Message"].astype(str).tolist()

                # Vectorize text
                vectorizer = CountVectorizer(
                    stop_words='english',
                    max_df=0.9,
                    min_df=2
                )
                X = vectorizer.fit_transform(messages)

                # Apply LDA topic modeling
                lda = LatentDirichletAllocation(
                    n_components=5,  # 5 topics
                    random_state=42,
                    learning_method='batch'
                )
                lda.fit(X)

                # Get topic keywords
                terms = vectorizer.get_feature_names_out()
                topics = []
                for idx, topic in enumerate(lda.components_):
                    top_terms = [terms[i] for i in topic.argsort()[-7:]]  # top 7 keywords
                    topics.append(", ".join(top_terms))

                # Display results
                st.subheader("🧩 Major Chat Themes Found :")
                for i, t in enumerate(topics):
                    st.markdown(f"**Theme {i + 1}:** {t}")

                # Optional — show topic distribution chart
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.bar(range(1, len(topics) + 1), lda.components_.sum(axis=1))
                ax.set_xlabel("Theme Number")
                ax.set_ylabel("Importance")
                ax.set_title("Distribution of Chat Themes")
                st.pyplot(fig)

            else:
                st.warning("⚠️ Not enough valid messages to extract topics.")
        else:
            st.info("📁 Please upload a WhatsApp chat file to detect chat themes.")

        # --- MOST ACTIVE CONNECTIONS (Without Graph) ---
        import streamlit as st
        import pandas as pd


        st.markdown('<h1 style="color: #efe9ae; font-size: 36px;">📊 Most Active Connections</h1>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Filter and clean chat data
            relation_df = df[df["User"] != "group_notification"]
            relation_df = relation_df.dropna(subset=["User", "Message"])

            # Build edges (message sequences between users)
            edges = []
            for i in range(len(relation_df) - 1):
                sender = relation_df.iloc[i]["User"]
                receiver = relation_df.iloc[i + 1]["User"]
                if sender != receiver:
                    edges.append((sender, receiver))

            if edges:
                # Count pair frequencies (bi-directional)
                from collections import Counter

                connection_counts = Counter()
                for u, v in edges:
                    connection_counts[(u, v)] += 1
                    connection_counts[(v, u)] += 1  # count both directions

                # Convert to DataFrame for sorting
                pairs = []
                for (u, v), count in connection_counts.items():
                    if u < v:  # avoid duplicates like (A,B) and (B,A)
                        pairs.append({"Person 1": u, "Person 2": v, "Interactions": count})
                result_df = pd.DataFrame(pairs).sort_values(by="Interactions", ascending=False)

                # Display Top 5
                st.subheader("💬 Top 5 Most Active Connections :")
                for _, row in result_df.head(5).iterrows():
                    st.markdown(f"""
                        <div style="
                            background:linear-gradient(90deg, #0f172a, #1e293b);
                            border:1px solid rgba(255,255,255,0.1);
                            border-radius:12px;
                            padding:12px 18px;
                            margin-bottom:10px;
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                            box-shadow:0 2px 8px rgba(0,0,0,0.3);">
                            <span style="font-weight:600; font-size:16px; color:#f1f5f9;">{row['Person 1']} ↔ {row['Person 2']}</span>
                            <span style="font-size:15px; color:#94a3b8;">{row['Interactions']} interactions</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Optional: show complete table
                with st.expander("View All Connections"):
                    st.dataframe(result_df.reset_index(drop=True))

            else:
                st.warning("⚠️ Not enough valid user messages to calculate connections.")
        else:
            st.info("📁 Please upload a WhatsApp chat file to view active connections.")


import streamlit as st

# Custom CSS for advanced animation and enhanced design
st.markdown(
    """
    <style>
        /* Custom gradient background animation */
        @keyframes gradientBg {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        /* Fade and zoom effect for text */
        @keyframes fadeInZoom {
            0% {
                opacity: 0;
                transform: translateY(30px) scale(0.6); /* Start smaller and lower */
            }
            100% {
                opacity: 1;
                transform: translateY(0) scale(1); /* End normal size */
            }
        }

        /* Pulse effect for the button */
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }

        /* Background Gradient Animation */
        body {
            background: linear-gradient(-45deg, #6a11cb, #2575fc);
            background-size: 400% 400%;
            animation: gradientBg 15s ease infinite;
        }

        /* Animated Text Styling (Neon Effect + Fade Zoom) */
        .animated-text {
            animation: fadeInZoom 3s ease-in-out;
            color: #ff5c8d;
            font-family: 'Roboto', sans-serif;
            font-size: 50px;
            text-align: center;
        }

        /* Button Styling with Pulse Effect */
        .animated-button {
            background: #ff6f61;
            color: #ffffff;
            font-size: 18px;
            padding: 15px 30px;
            text-align: center;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0px 10px 20px rgba(255, 102, 102, 0.4);
            animation: pulse 2s infinite;
            transition: transform 0.2s;
        }

        /* Hover effect for the button */
        .animated-button:hover {
            transform: translateY(-5px); /* Makes the button slightly lift up */
        }

        /* Additional Custom Animation */
        .animated-footer {
            text-align: center;
            margin-top: 20px;
            font-size: 18px;
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True
)

# Main Content
st.markdown('<h1 class="animated-text">🚀 Crunch Your Chats Effortlessly With "ChatIQ" !</h1>', unsafe_allow_html=True)

if st.button("Help"):
    st.info(
        "To Export A WhatsApp Chat :\n1. Open The Chat\n2. Go To Options -> More -> Export Chat -> Without Media"
    )











