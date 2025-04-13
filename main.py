import streamlit as st
import regression  
import clustering
import neural_networks
import llm_rag



# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


# Function to change page
def go_to(page_name):
    st.session_state.page = page_name

# 🌟 Page 1: Welcome Page
if st.session_state.page == "home":
    st.markdown("""
        <style>
            .big-title {
                font-size: 80px;
                color: #4CAF50;
                font-weight: 900;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 20px;
            }
            .description {
                font-size: 28px;
                text-align: center;
                color: #555555;
                font-weight: 500;
                margin-bottom: 40px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
            }
            .stButton > button {
                display: block;
                margin: 0 auto;
                padding: 0.8em 2em;
                font-size: 18px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>

        <div class="big-title">🎓 AI Scholar Hub</div>
        <div class="description">A hands-on platform showcasing the power of machine learning and artificial intelligence through interactive demos and smart models.</div>
    """, unsafe_allow_html=True)

    if st.button("Let's Get Started"):
        go_to("auth")
        st.rerun()


# 🔐 Page 2: Sign In or Sign Up
elif st.session_state.page == "auth":
    st.markdown("""
        <style>
            .auth-title {
                font-size: 48px;
                color: #4CAF50;
                font-weight: 700;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 20px;
            }
            .stButton > button {
                display: block;
                margin: 0 auto;
                padding: 0.8em 2em;
                font-size: 18px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>

        <div class="auth-title">Welcome to AI Scholar Hub</div>
        <div style="text-align: center; color: #555555; font-size: 24px; margin-bottom: 40px;">
            Please choose one of the following options to proceed.
        </div>
        <div style="text-align: center; font-size: 48px; margin-bottom: 40px;">😊</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sign In"):
            go_to("signin")
            st.rerun()


    with col2:
        if st.button("Sign Up"):
            go_to("signup")
            st.rerun()


# ✍️ Page 3: Sign Up
elif st.session_state.page == "signup":
    st.markdown("""
        <style>
            .auth-title {
                font-size: 48px;
                color: #4CAF50;
                font-weight: 700;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 10px;
            }
            .auth-description {
                text-align: center;
                font-size: 20px;
                color: #555555;
                margin-bottom: 30px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                padding: 0.6em 2em;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
        <div class="auth-title">Create Your Account</div>
        <div class="auth-description">Join the AI Scholar Hub and start exploring!</div>
    """, unsafe_allow_html=True)

    with st.form("signup_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")

    if submit:
        if not full_name or not email or not username or not password or not confirm_password:
            st.error("Please fill in all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            # Save user name and go to welcome page
            st.session_state.user_name = full_name.split()[0]  # Just first name
            st.session_state.page = "welcome"
            st.rerun()

# for after pressing the sign in button
elif st.session_state.page == "signin":
    st.markdown("""
        <style>
            .login-title {
                font-size: 48px;
                color: #4CAF50;
                font-weight: 700;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 10px;
            }
            .login-description {
                text-align: center;
                font-size: 20px;
                color: #555555;
                margin-bottom: 30px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                padding: 0.6em 2em;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
        <div class="login-title">Sign In</div>
        <div class="login-description">Welcome back! Please log in to continue.</div>
    """, unsafe_allow_html=True)

    with st.form("signin_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")

    if submit:
        # For now, we'll accept any non-empty input
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            # Simulate login success
            st.session_state.user_name = username  # Display username on welcome page
            st.session_state.page = "welcome"
            st.rerun()




# for after the sign up
elif st.session_state.page == "welcome":
    user_name = st.session_state.get("user_name", "User")

    st.markdown(f"""
        <style>
            .welcome-title {{
                font-size: 42px;
                color: #4CAF50;
                font-weight: 800;
                text-align: center;
                margin-top: 30px;
                margin-bottom: 20px;
            }}
            .explore-btn > button {{
                display: block;
                margin: 0 auto;
                padding: 0.8em 2em;
                font-size: 18px;
                background-color: #4CAF50;  /* Green background */
                color: white;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }}
            .explore-btn > button:hover {{
                background-color: #45a049;  /* Darker green on hover */
            }}
            .stButton > button {{
                display: block;
                margin: 0 auto;
                padding: 0.8em 2em;
                font-size: 18px;
                background-color: #4CAF50;  /* Green background for sign-in and sign-up buttons */
                color: white;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }}
            .stButton > button:hover {{
                background-color: #45a049;  /* Darker green on hover */
            }}
        </style>

        <div class="welcome-title">👋 Welcome, {user_name}!</div>
        <div style="text-align:center; font-size: 24px; margin-bottom: 40px;">
            Ready to explore the power of AI?
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        # Create a container for the button to apply CSS
        with st.markdown('<div class="explore-btn">', unsafe_allow_html=True):
            if st.button("Explore AI Services"):
                st.session_state.page = "services"
                st.rerun()  # Rerun to reload the page content

   

# this is for services

elif st.session_state.page == "services":
    # Display styles
    st.markdown("""
        <style>
            .service-title {
                font-size: 38px;
                color: #4CAF50;
                font-weight: bold;
                text-align: center;
                margin-bottom: 30px;
            }
            .stButton > button {
                background-color: white !important;
                color: #333333 !important;
                border: 2px solid #4CAF50;
                border-radius: 12px;
                font-size: 22px;
                padding: 20px 40px;
                width: 100%;
                height: 100px;
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #4CAF50 !important;
                color: white !important;
                cursor: pointer;
            }
                
            .home-btn > button {
                display: block;
                margin: 30px auto 0 auto;
                padding: 0.6em 1.5em;
                font-size: 25px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                transition: background-color 0.3s ease;
            }
            .home-btn > button:hover {
                background-color: #45a049;
            }   
        </style>
    """, unsafe_allow_html=True)

    # Display service title
    st.markdown('<div class="service-title">🧠 AI Services</div>', unsafe_allow_html=True)

    # Display the services grid with clickable buttons for each service
    col1, col2 = st.columns(2)

    with col1:
        if st.button('✅ Regression', key='regression_button'):
            st.session_state.page = "regression"
            st.rerun()


    with col2:
        if st.button('🔄 Clustering', key='clustering_button'):
            st.session_state.page = "clustering"
            st.rerun()


    col3, col4 = st.columns(2)

    with col3:
        if st.button('🧠 Neural Networks', key='nn_button'):
            st.session_state.page = "neural_networks"
            st.rerun()


    with col4:
        if st.button('💬 Large Language Model', key='llm_button'):
            st.session_state.page = "large_language_model"
            st.rerun()

    # Back to Home button at bottom
    with st.container():
        st.markdown('<div class="home-btn">', unsafe_allow_html=True)
        if st.button("⬅️ Back to Home", key='back_to_home'):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.page == "regression":
    regression.run()  # This calls the run function from regression.py to display the regression analysis page


elif st.session_state.page == "clustering":
    clustering.run()

elif st.session_state.page == "neural_networks":
    neural_networks.run()

elif st.session_state.page == "large_language_model":
    llm_rag.run()
