# Part 1: Warmup — Cloud Concepts

# Cloud Concepts Question 1
# Cloud computing uses a pay-as-you-go model. Instead of buying and maintaining
# your own servers, you rent the resources you need and pay for what you use.

# Cloud Concepts Question 2
# Vertical scaling means making one machine more powerful, like adding more RAM or
# getting a better GPU. Horizontal scaling means adding more machines to handle
# the work.

# Web app: Horizontal scaling because more machines can help handle the increase
# from 1,000 users to 100,000 users.

# ML model: Vertical scaling because they need a more powerful machine with a
# better GPU and more RAM.

# Data pipeline: Horizontal scaling because the files can be split between
# multiple machines.

# Cloud Concepts Question 3

# Gmail - SaaS: You use the application while the provider manages it.

# AWS - IaaS: You rent a virtual machine and manage the
# operating system and software yourself.

# AWS S3 - Object storage: It is used to store files and access them by a key.

# GitHub Codespaces - Not classified in the lesson.

# Snowflake - Managed data platform: It is a managed platform for working with
# data.

# Supabase - BaaS: It provides things like a database, authentication, and
# storage for applications.

# IaaS: You rent basic computing resources and manage the operating system and
# software yourself.
# Example: Azure Virtual Machines.

# PaaS: The provider manages the infrastructure not you and you mainly manage your code
# and application.
# Example: AWS.

# SaaS: You use an application that the provider manages for you.
# Example: Gmail.

# Cloud Concepts Question 4
# A managed data platform like Snowflake or Databricks gives you tools for
# working with data without having to set everything up yourself. You gain
# easier setup but give up some flexibility.

# Cloud Concepts Question 5
# The cloud may not be the best choice when your data and computing needs are
# small enough to fit on one machine.
# It may also not be worth it when setting up and learning the cloud takes more
# time than the project requires.


# Part 2: Warmup — Cloud Landscape

# Cloud Landscape Question 1

# AWS - Has the largest variety of cloud services and is often used by large
# companies, startups, and organizations with engineering teams.

# GCP - Is especially strong in data and machine learning and is often used for
# analytics and ML projects.

# Azure - Is strong for businesses and government because it works well with
# Microsoft's other products.

# Cloud Landscape Question 2

# Access - Supabase was easier for students to access and had a free tier that
# was enough for the course.

# Pedagogical fit - Supabase uses a relational database, which teaches useful
# skills like working with tables and querying data.

# Pipeline coherence - Supabase made it easier to organize the raw and enriched
# data for the ETL pipeline.

# Reflection - I think I should choose a cloud tool based on how easy it is to
# access, how well it fits the project, and what skills I can learn from it.

# Cloud Landscape Question 3

# 1. Object storage - AWS S3
# It can store large amounts of image files and let you access them by key.

# 2. ML platform - AWS SageMaker
# It can be used to train machine learning models with a GPU.

# 3. Serverless compute - AWS Lambda
# It can run the API without me having to manage the servers.

# 4. LLM API - Azure OpenAI
# It lets me send information to a large language model and get a text response.

# Cloud Landscape Question 4
# I could make a weather data project using Supabase for the database and
# BigQuery for analyzing the data. I could also use an LLM API to process some
# of the data.

# Using one provider could make things simpler, but I would lose the ability to
# choose different services that might be better for specific parts of the
# project.