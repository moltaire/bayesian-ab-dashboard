# Bayesian A/B Dashboard

A small interactive tool for updating Beta-Binomial posteriors as you collect data from an A/B test. Set your priors, feed in successes and failures, and watch the posteriors evolve.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bayesian-ab-dashboard.streamlit.app/)

## What it does

- Shows posterior distributions for two variants (A and B)
- Tracks how the posterior mean shifts across update steps
- Gives you P(B > A) and expected loss via Monte Carlo

## Usage

Adjust the priors in the sidebar, enter observed successes/failures, and hit **Update posteriors**. Use **Reset to priors** to start fresh from the current prior settings, or **Default priors** to go back to the original values.
