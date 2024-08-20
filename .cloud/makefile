select-dev:
	terraform workspace select dev

select-prod:
	terraform workspace select prod

plan:
	terraform plan -var-file=input.tfvars -auto-approve

destroy-instance:
	terraform destroy -var-file=input.tfvars -target module.ec2_instance

destroy:
	terraform destroy -var-file=input.tfvars

apply:
	terraform apply -var-file=input.tfvars
