build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION}

run_train:
	python -c 'from aerosound.interface.main import train; train()'

run_evaluate:
	python -c 'from aerosound.interface.main import evaluate; evaluate()'
