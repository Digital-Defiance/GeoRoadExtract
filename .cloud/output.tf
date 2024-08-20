
output "instance_ip" {
  value = resource.aws_eip.lb.public_ip
}

output "ssh_command" {
  value = "ssh -i id_rsa ec2-user@${resource.aws_eip.lb.public_ip}"
}
