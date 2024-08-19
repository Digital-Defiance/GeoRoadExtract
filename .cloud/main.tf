terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-south-1"
}

resource "aws_eip" "lb" {
  instance   = module.ec2_instance.spot_instance_id
  domain     = "vpc"
  depends_on = [module.ec2_instance]
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "SRE-VPC"
  cidr = "10.0.0.0/16"

  azs             = ["eu-south-1b"]
  private_subnets = []
  public_subnets  = ["10.0.101.0/24"]

  enable_nat_gateway = false
  enable_vpn_gateway = false

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}

variable "public_ssh_key" {
  type        = string
  description = "The public ssh key" 
  
}

resource "aws_key_pair" "default" {
  key_name   = "SRE-KEY-PAIR"
  public_key = var.public_ssh_key
}

module "ssh_security_group" {
  source              = "terraform-aws-modules/security-group/aws//modules/ssh"
  version             = "~> 5.0"
  name                = "ssh-security-group"
  vpc_id              = module.vpc.vpc_id
  ingress_cidr_blocks = ["0.0.0.0/0"]
}


module "ec2_instance" {
  source = "terraform-aws-modules/ec2-instance/aws"

  name                 = "SRE-GPU-INSTANCE"
  create_spot_instance = true

  instance_type = "g4dn.xlarge"

  key_name               = resource.aws_key_pair.default.key_name
  monitoring             = true
  vpc_security_group_ids = [module.ssh_security_group.security_group_id]
  subnet_id              = module.vpc.public_subnets[0]

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}



