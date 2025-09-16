pipeline {
    agent any
    
    stages {
        stage('Cloning From Github repo') {
            steps {
                script {
                    echo 'Cloning From Githubb repoo...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'mlops-github-token', url: 'https://github.com/jiteshvk13/MLOPwithjenkins.git']])
                }
            }
        }
    }
}