pipeline{
    agent any
    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = "striped-torus-479904-v4"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins.......'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_jen', url: 'https://github.com/sumithgs/mlproject.git']])
                }
            }
        }
        stage('Setting up virtual environment and installing dependencies') {
            steps {
                script {
                    echo 'Setting up virtual environment and installing dependencies.......'
                    sh '''
                    # Ensure Python 3.11 is used
                    PYTHON_BIN=$(which python3.11)
                    if [ -z "$PYTHON_BIN" ]; then
                        echo "Python 3.11 not found!"
                        exit 1
                    fi

                    # Create venv with Python 3.11
                    $PYTHON_BIN -m venv ${VENV_DIR}

                    # Activate and install dependencies
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

                stage('Run Training Pipeline') {
                    steps {
                        script {
                            echo 'Running Training Pipeline...'
                            sh '''
                            . ${VENV_DIR}/bin/activate
                            python3 -m src.pipeline.train_pipeline
                            '''
                        }
                    }
                }
        // stage('Building and Pushing Docker Image to GCR'){
        //     steps{
        //         withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
        //             script{
        //                 echo 'Building and Pushing Docker Image to GCR.............'
        //                 sh '''
        //                 export PATH=$PATH:${GCLOUD_PATH}

        //                 gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
        //                 gcloud config set project ${GCP_PROJECT}
        //                 gcloud auth configure-docker --quiet

        //                 docker build --platform linux/amd64 -t gcr.io/${GCP_PROJECT}/ml-project:latest .
        //                 docker push gcr.io/${GCP_PROJECT}/ml-project:latest
        //                 '''
        //             }
        //         }
        //     }
        // }

        // stage('Deploy to Google Cloud Run'){
        //     steps{
        //         withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
        //             script{
        //                 echo 'Deploy to Google Cloud Run.............'
        //                 sh '''
        //                 export PATH=$PATH:${GCLOUD_PATH}


        //                 gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

        //                 gcloud config set project ${GCP_PROJECT}

        //                 gcloud run deploy ml-project \
        //                     --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
        //                     --platform=managed \
        //                     --region=us-central1 \
        //                     --allow-unauthenticated
                            
        //                 '''
        //             }
        //         }
        //     }
        // }

    }
}