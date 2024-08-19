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

resource "aws_key_pair" "default" {
  key_name   = "SRE-KEY-PAIR"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCYuMXcQUrMCI08/EU3PL+qEiipviiRKelbNpfdYVb0x8JXEWU6yjEnwZgLxigxClGIxgHFDPsBqYNjJtzKlR8/w5x8DR/Mbdt0b0y4QDwaPTwwa/Onav6rqJiBi50uq+DhntY4KeJRjKZfUHGcWsuvh4FZKx9cfyuby49Z7vxQ6Nz7QhiROHVlIcbdgT9V1QXQq/t0J5A4aFYeoG//vzdEzLeSilxiXz4gyx+K+CmI6d67+eSwd1k0vaRaLfVt26znX6PSqtUNHPLl5wLAhUmMiYHjQwx6W4LXDj6EolPzekLB4xvh6rBnkLYH5FXuxdERrr5pjapa6dmNyfoHNw2lOvqPnmenun9IVvzhL6TvJmgdfZgXdeGH76zgDY9yPlryb/YhbkYW22ZuMZyK5OZC7MFBUIBcNv5oHTS6U62o/ntBYiAQi6DTICVpVZaQnYUtB1CyLuaLjxKp/9yoT8edRsT1Ue0RomaKxKZHBsGX4wJq32tXND9WmTl42RlWp2a+fxRlGFPTbpLicBURKWhiQ/mIraSdtUjyJvHS83HzHBbcTxEBg3yCOcLhC2IpKtBRddCv72GkMGOJZA6VhwLdJ1jEJ/P5F07/0PjG81dlrbCHHbknTGFdgOd4fxjf4PbLaN7j39IedZpgOzO1QltM53Z1SDh9wi7GQmJ91vYfVw== rui@rui-ThinkPad-E15"
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



