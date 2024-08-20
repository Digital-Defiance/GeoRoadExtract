

terraform {
  backend "s3" {
    key    = "env"
    bucket = "digitaldefiance-terraform-backend"
    region = "eu-south-1"
  }

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

resource "aws_eip_association" "eip_assoc" {
  count = var.create_instance ? 1 : 0
  instance_id   = module.ec2_instance.spot_instance_id
  allocation_id = resource.aws_eip.lb.id
}


resource "aws_eip" "lb" {
  tags = {
    Name = "SRE-IP-${terraform.workspace}"
  }
  domain     = "vpc"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "SRE-VPC-${terraform.workspace}"
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

resource "aws_key_pair" "default" {
  key_name   = "SRE-KEY-PAIR-${terraform.workspace}"
  public_key = var.public_ssh_key
}

module "ssh_security_group" {
  name                = "ssh-security-group-${terraform.workspace}"
  source              = "terraform-aws-modules/security-group/aws//modules/ssh"
  version             = "~> 5.0"
  vpc_id              = module.vpc.vpc_id
  ingress_cidr_blocks = ["0.0.0.0/0"]
}

module "ec2_instance" {
  ami = "ami-0c261f3985286d47f"
  source = "terraform-aws-modules/ec2-instance/aws"
  user_data = <<EOF
#!/bin/bash
sudo yum update -y
sudo yum -y install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo chmod 666 /var/run/docker.sock
docker version
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum-config-manager --enable nvidia-container-toolkit-experimental
sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
  EOF

  name                 = "SRE-GPU-INSTANCE-${terraform.workspace}"
  create_spot_instance = true
  create               = var.create_instance

  instance_type = "g4dn.xlarge"

  key_name               = resource.aws_key_pair.default.key_name
  monitoring             = true
  vpc_security_group_ids = [module.ssh_security_group.security_group_id]
  subnet_id              = module.vpc.public_subnets[0]

  instance_tags = {
    Name = "SRE-GPU-INSTANCE-${terraform.workspace}"
  }

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}



