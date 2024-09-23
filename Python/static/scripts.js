
        function showPredictionForm() {
            document.getElementById('prediction-form').style.display = 'block';
            document.getElementById('reason-form').style.display = 'none';
            document.getElementById('cluster-form').style.display = 'none';
        }

        function showReasonForm() {
            document.getElementById('reason-form').style.display = 'block';
            document.getElementById('prediction-form').style.display = 'none';
            document.getElementById('cluster-form').style.display = 'none';
        }

        function showClusterForm() {
            document.getElementById('cluster-form').style.display = 'block';
            document.getElementById('prediction-form').style.display = 'none';
            document.getElementById('reason-form').style.display = 'none';
        }