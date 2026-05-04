terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "ecommerce-recommender"
}

# S3 bucket for trained models
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Kinesis stream for clickstream events
resource "aws_kinesis_stream" "clickstream" {
  name             = "${var.project_name}-clickstream"
  shard_count      = 2
  retention_period = 24
}

# DynamoDB table for user state
resource "aws_dynamodb_table" "user_state" {
  name         = "${var.project_name}-user-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# OpenSearch domain
resource "aws_opensearch_domain" "products" {
  domain_name    = "${var.project_name}-products"
  engine_version = "OpenSearch_2.11"

  cluster_config {
    instance_type  = "t3.small.search"
    instance_count = 2
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 20
    volume_type = "gp3"
  }
}

# ECS cluster + Fargate service (skeleton)
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
}

output "kinesis_stream_arn" {
  value = aws_kinesis_stream.clickstream.arn
}

output "user_state_table_name" {
  value = aws_dynamodb_table.user_state.name
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.products.endpoint
}
