import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModel


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AspectSense Admin",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_DIR = "absa_model"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# =========================================================
# MODEL
# =========================================================

class ABSAModel(torch.nn.Module):

    def __init__(self, model_dir):

        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            model_dir
        )

        hidden_size = (
            self.transformer.config.hidden_size
        )

        self.dropout = torch.nn.Dropout(0.2)

        self.classifier = torch.nn.Linear(
            hidden_size,
            3
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = (
            outputs.last_hidden_state[:, 0, :]
        )

        cls_output = self.dropout(
            cls_output
        )

        return self.classifier(
            cls_output
        )


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = ABSAModel(
        MODEL_DIR
    )

    weights = torch.load(
        f"{MODEL_DIR}/classifier.pt",
        map_location=DEVICE
    )

    model.classifier.load_state_dict(
        weights
    )

    model.to(DEVICE)

    model.eval()

    return tokenizer, model


try:

    tokenizer, model = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    model_error = str(e)


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:

    st.session_state.results = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧠 AspectSense")

st.sidebar.caption(
    "Aspect-Based Sentiment Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Admin Dashboard",
        "🔍 Analyze Review",
        "📁 Batch Predictions",
        "⚙️ Model Info"
    ]
)

st.sidebar.divider()

st.sidebar.write("### System")

if model_loaded:

    st.sidebar.success("Model Loaded")

else:

    st.sidebar.error("Model Failed")

    st.sidebar.caption(
        model_error
    )

st.sidebar.write(
    f"Device: `{DEVICE}`"
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🧠 AspectSense")

st.caption(
    "AI-powered Aspect-Based Sentiment Analysis Dashboard"
)

st.divider()


# =========================================================
# ADMIN DASHBOARD
# =========================================================

if page == "📊 Admin Dashboard":

    st.header("📊 Admin Dashboard")

    st.write(
        "Monitor sentiment predictions and aspect-level insights."
    )

    results = st.session_state.results


    # -----------------------------------------------------
    # NO DATA YET
    # -----------------------------------------------------

    if results is None:

        st.info(
            "No predictions available yet. "
            "Go to **Batch Predictions** and run the test dataset."
        )

        st.subheader("Hackathon Dataset")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Training Samples",
            "8,000"
        )

        c2.metric(
            "Test Samples",
            "2,000"
        )

        c3.metric(
            "Sentiment Classes",
            "3"
        )

        c4.metric(
            "Model",
            "DistilBERT"
        )

        st.divider()

        st.subheader(
            "Dashboard will show"
        )

        a, b, c = st.columns(3)

        a.success(
            "🟢 Positive\n\n"
            "Number and percentage of positive predictions"
        )

        b.warning(
            "🟡 Neutral\n\n"
            "Number and percentage of neutral predictions"
        )

        c.error(
            "🔴 Negative\n\n"
            "Number and percentage of negative predictions"
        )


    # -----------------------------------------------------
    # SHOW DASHBOARD
    # -----------------------------------------------------

    else:

        total = len(results)

        positive = int(
            (
                results["prediction"]
                == "Positive"
            ).sum()
        )

        neutral = int(
            (
                results["prediction"]
                == "Neutral"
            ).sum()
        )

        negative = int(
            (
                results["prediction"]
                == "Negative"
            ).sum()
        )


        positive_pct = (
            positive / total * 100
        )

        neutral_pct = (
            neutral / total * 100
        )

        negative_pct = (
            negative / total * 100
        )


        # =================================================
        # TOP KPI CARDS
        # =================================================

        st.subheader(
            "📌 Sentiment Overview"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Reviews",
            f"{total:,}"
        )

        c2.metric(
            "🟢 Positive",
            f"{positive:,}",
            f"{positive_pct:.1f}%"
        )

        c3.metric(
            "🟡 Neutral",
            f"{neutral:,}",
            f"{neutral_pct:.1f}%"
        )

        c4.metric(
            "🔴 Negative",
            f"{negative:,}",
            f"{negative_pct:.1f}%"
        )


        st.divider()


        # =================================================
        # SENTIMENT CHARTS
        # =================================================

        st.subheader(
            "📈 Sentiment Analytics"
        )

        chart_data = pd.DataFrame({

            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],

            "Count": [
                positive,
                neutral,
                negative
            ]

        })


        left, right = st.columns(2)


        # PIE CHART

        with left:

            st.write(
                "### 🍩 Sentiment Distribution"
            )

            pie = px.pie(
                chart_data,
                names="Sentiment",
                values="Count",
                hole=0.55
            )

            pie.update_layout(
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )


        # BAR CHART

        with right:

            st.write(
                "### 📊 Sentiment Comparison"
            )

            bar = px.bar(
                chart_data,
                x="Sentiment",
                y="Count",
                text="Count"
            )

            bar.update_traces(
                textposition="outside"
            )

            bar.update_layout(
                height=450,
                yaxis_title="Number of Reviews",
                xaxis_title=""
            )

            st.plotly_chart(
                bar,
                use_container_width=True
            )


        # =================================================
        # PERCENTAGE
        # =================================================

        st.subheader(
            "📊 Sentiment Percentage"
        )

        percentage_data = pd.DataFrame({

            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],

            "Percentage": [
                positive_pct,
                neutral_pct,
                negative_pct
            ]

        })


        percentage_chart = px.bar(
            percentage_data,
            x="Sentiment",
            y="Percentage",
            text="Percentage"
        )

        percentage_chart.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        percentage_chart.update_layout(
            yaxis_title="Percentage (%)",
            xaxis_title="",
            yaxis_range=[0, 100],
            height=400
        )

        st.plotly_chart(
            percentage_chart,
            use_container_width=True
        )


        # =================================================
        # TOP ASPECTS
        # =================================================

        st.divider()

        st.subheader(
            "🎯 Top 10 Aspects"
        )

        aspect_counts = (
            results["aspect_term"]
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )

        aspect_counts.columns = [
            "Aspect",
            "Reviews"
        ]


        aspect_chart = px.bar(
            aspect_counts,
            x="Reviews",
            y="Aspect",
            orientation="h",
            text="Reviews"
        )

        aspect_chart.update_traces(
            textposition="outside"
        )

        aspect_chart.update_layout(
            height=450,
            xaxis_title="Number of Reviews",
            yaxis_title=""
        )

        st.plotly_chart(
            aspect_chart,
            use_container_width=True
        )


        # =================================================
        # ASPECT-WISE SENTIMENT
        # =================================================

        st.subheader(
            "🔎 Aspect-wise Sentiment"
        )

        aspect_sentiment = (
            results
            .groupby(
                [
                    "aspect_term",
                    "prediction"
                ]
            )
            .size()
            .reset_index(
                name="Count"
            )
        )


        top_aspects = (
            results["aspect_term"]
            .astype(str)
            .value_counts()
            .head(8)
            .index
        )


        aspect_sentiment = (
            aspect_sentiment[
                aspect_sentiment[
                    "aspect_term"
                ].isin(
                    top_aspects
                )
            ]
        )


        grouped_chart = px.bar(
            aspect_sentiment,
            x="aspect_term",
            y="Count",
            color="prediction",
            barmode="group"
        )

        grouped_chart.update_layout(
            height=500,
            xaxis_title="Aspect",
            yaxis_title="Reviews"
        )

        st.plotly_chart(
            grouped_chart,
            use_container_width=True
        )


        # =================================================
        # DATA TABLE
        # =================================================

        st.subheader(
            "📋 Prediction Preview"
        )

        st.dataframe(
            results[
                [
                    "id",
                    "text",
                    "aspect_term",
                    "prediction"
                ]
            ].head(20),
            use_container_width=True
        )


# =========================================================
# ANALYZE ONE REVIEW
# =========================================================

elif page == "🔍 Analyze Review":

    st.header(
        "🔍 Analyze a Review"
    )

    review = st.text_area(
        "Review Text",
        placeholder=(
            "The camera is excellent "
            "but the battery life is poor."
        ),
        height=150
    )

    aspect = st.text_input(
        "Aspect Term",
        placeholder="camera"
    )


    if st.button(
        "🚀 Analyze Sentiment",
        use_container_width=True
    ):

        if not review or not aspect:

            st.warning(
                "Please enter both review and aspect."
            )

        else:

            encoding = tokenizer(
                aspect,
                review,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )


            input_ids = encoding[
                "input_ids"
            ].to(DEVICE)

            attention_mask = encoding[
                "attention_mask"
            ].to(DEVICE)


            with torch.no_grad():

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[0]


            prediction = torch.argmax(
                probabilities
            ).item()


            sentiment = LABEL_MAP[
                prediction
            ]


            confidence = (
                probabilities[
                    prediction
                ].item()
                * 100
            )


            st.success(
                f"Aspect: **{aspect}**  |  "
                f"Prediction: **{sentiment}**  |  "
                f"Confidence: **{confidence:.2f}%**"
            )


            c1, c2, c3 = st.columns(3)


            c1.metric(
                "🔴 Negative",
                f"{probabilities[0].item()*100:.1f}%"
            )

            c2.metric(
                "🟡 Neutral",
                f"{probabilities[1].item()*100:.1f}%"
            )

            c3.metric(
                "🟢 Positive",
                f"{probabilities[2].item()*100:.1f}%"
            )


            confidence_df = pd.DataFrame({

                "Sentiment": [
                    "Negative",
                    "Neutral",
                    "Positive"
                ],

                "Confidence": [
                    probabilities[0].item() * 100,
                    probabilities[1].item() * 100,
                    probabilities[2].item() * 100
                ]

            })


            fig = px.bar(
                confidence_df,
                x="Sentiment",
                y="Confidence",
                text="Confidence"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_range=[0, 110],
                yaxis_title="Confidence (%)",
                xaxis_title=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# BATCH PREDICTIONS
# =========================================================

elif page == "📁 Batch Predictions":

    st.header(
        "📁 Batch Predictions"
    )

    st.write(
        "Upload the hackathon test.csv and generate "
        "predictions for all test samples."
    )


    uploaded_file = st.file_uploader(
        "Upload test.csv",
        type=["csv"]
    )


    if uploaded_file is not None:

        test_df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Loaded {len(test_df):,} rows"
        )

        st.dataframe(
            test_df.head(10),
            use_container_width=True
        )


        if st.button(
            "🚀 Generate Predictions",
            use_container_width=True
        ):

            predictions = []

            progress = st.progress(0)

            status = st.empty()


            # Process in batches
            batch_size = 32

            for start in range(
                0,
                len(test_df),
                batch_size
            ):

                end = min(
                    start + batch_size,
                    len(test_df)
                )


                aspects = (
                    test_df[
                        "aspect_term"
                    ]
                    .astype(str)
                    .iloc[start:end]
                    .tolist()
                )


                texts = (
                    test_df[
                        "text"
                    ]
                    .astype(str)
                    .iloc[start:end]
                    .tolist()
                )


                encoding = tokenizer(
                    aspects,
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )


                input_ids = encoding[
                    "input_ids"
                ].to(DEVICE)

                attention_mask = encoding[
                    "attention_mask"
                ].to(DEVICE)


                with torch.no_grad():

                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    batch_predictions = (
                        torch.argmax(
                            logits,
                            dim=1
                        )
                        .cpu()
                        .tolist()
                    )


                predictions.extend(
                    [
                        LABEL_MAP[p]
                        for p in batch_predictions
                    ]
                )


                progress.progress(
                    end / len(test_df)
                )

                status.write(
                    f"Analyzing "
                    f"{end:,} / "
                    f"{len(test_df):,}"
                )


            result = test_df.copy()

            result[
                "prediction"
            ] = predictions


            st.session_state.results = result


            st.success(
                f"🎉 {len(result):,} predictions generated!"
            )


            submission = pd.DataFrame({

                "id": result["id"],

                "prediction": result[
                    "prediction"
                ]

            })


            csv_data = submission.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                "⬇️ Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# MODEL INFO
# =========================================================

elif page == "⚙️ Model Info":

    st.header(
        "⚙️ Model Information"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Architecture",
        "DistilBERT"
    )

    c2.metric(
        "Training Samples",
        "8,000"
    )

    c3.metric(
        "Test Samples",
        "2,000"
    )

    c4.metric(
        "Classes",
        "3"
    )


    st.divider()


    st.subheader(
        "🔄 Model Pipeline"
    )


    st.write(
        "Review + Aspect"
    )

    st.write("↓")

    st.write(
        "Tokenizer"
    )

    st.write("↓")

    st.write(
        "DistilBERT"
    )

    st.write("↓")

    st.write(
        "3-Class Classification Layer"
    )

    st.write("↓")

    st.write(
        "Negative / Neutral / Positive"
    )


    st.info(
        "The Positive, Neutral and Negative counts "
        "in the Admin Dashboard are model predictions "
        "for the test dataset. They are not ground-truth labels."
    )