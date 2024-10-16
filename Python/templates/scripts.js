
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

function showForm(formId) {
    // Ẩn tất cả các form dự đoán
    document.querySelectorAll('.predict-form').forEach(form => form.classList.add('hidden'));
    // Hiện form dự đoán được chọn
    document.getElementById(formId).classList.remove('hidden');
}



function predictCluster(event) {
    event.preventDefault(); // Ngăn chặn gửi form mặc định

    // Lấy dữ liệu từ form
    const formData = new FormData(event.target);

    fetch('/predict_cluster', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hiển thị kết quả dưới form
        const resultDiv = document.getElementById('cluster_prediction_result');
        resultDiv.innerHTML = `<h2>Kết Quả Dự Đoán Cụm</h2><p>${data.prediction_cluster}</p>`;
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('cluster_prediction_result');
        resultDiv.innerHTML = `<h2>Lỗi</h2><p>${error}</p>`;
    });
}

    const selectBox1 = document.getElementById('select_box_1');
    const displayText = {
        col1: "\n",
        col2: "\n",
        col3: "\n"
    };

    // Cập nhật văn bản cho display_text
    function updateDisplayText() {
        document.getElementById('col1').innerHTML = `${displayText.col1.replace(/\n/g, '<br>')}`;
        document.getElementById('col2').innerHTML = `${displayText.col2.replace(/\n/g, '<br>')}`;
        document.getElementById('col3').innerHTML = `${displayText.col3.replace(/\n/g, '<br>')}`;
    }

    selectBox1.addEventListener('change', function() {
        const selectedValue = this.value;

        if (selectedValue) {
            // Cập nhật văn bản theo tùy chọn cụ thể
            switch (selectedValue) {
                case 'nguyen_nhan':
                    displayText.col1 = "\n- Từ năm 2022 đến nay, nguyên nhân chính gây tai nạn giao thông ở TPHCM bao gồm " + 
                                    "va chạm giữa các phương tiện (chiếm tỷ lệ lớn, đặc biệt là năm 2023), " + 
                                    "không tuân thủ quy định giao thông (đặc biệt phổ biến trong năm 2022 nhưng giảm vào 2023), " + 
                                    "mất lái, không chú ý quan sát, và say xỉn khi lái xe. \n \n" +
                                    "- Nguyên nhân gây tai nạn giao thông năm 2023 bao gồm: " + 
                                    "Va chạm giữa các phương tiện chiếm tỷ lệ cao nhất với 37%, " + 
                                    "không tuân thủ quy định giảm nhờ các biện pháp siết chặt giao thông. " + 
                                    "Say xỉn chiếm 7%, và các vấn đề như mất lái, không chú ý quan sát cũng tiếp tục tồn tại.\n\n" +
                                    "- Nguyên nhân gây tai nạn giao thông năm 2022 tập trung vào việc " + 
                                    "không tuân thủ quy định giao thông chiếm 27% và va chạm giữa các phương tiện chiếm 26%. " + 
                                    "Mất lái và không chú ý quan sát cũng là các nguyên nhân phổ biến, lần lượt chiếm 15% và 10%.\n\n";
                    displayText.col2 =  
                                    "\n**Nguyên nhân tai nạn năm 2023:**\n\n" +
                                    "- Va chạm giữa các phương tiện: Chiếm tỉ lệ cao nhất với 37% tổng vụ án.\n" +
                                    "- Không tuân thủ quy định giao thông: Chiếm khoảng 18% tổng vụ án.\n" +
                                    "- Mất lái: Chiếm khoảng 19% tổng vụ án.\n" +
                                    "- Không chú ý quan sát: Chiếm khoảng 7% tổng vụ án.\n\n" +
                                    "**Địa điểm trọng điểm:**\n\n" +
                                    "- Thủ Đức: Là quận có số vụ tai nạn cao nhất với 12,42% tổng vụ án.\n" +
                                    "- Quốc Lộ 1 và Xa lộ Hà Nội: Các con đường lớn với mật độ giao thông cao, thường xuyên xảy ra tai nạn.\n" +
                                    "- Đường Nguyễn Văn Linh: Nơi có nhiều va chạm do sự lưu thông đông đúc.";

                    displayText.col3 = 
                                    "\n**Nguyên nhân tai nạn năm 2022:**\n\n" +
                                    "- Không tuân thủ quy định giao thông: Chiếm khoảng 27% tổng vụ án.\n" +
                                    "- Va chạm giữa các phương tiện: Chiếm tỉ lệ cao nhất với 26% tổng vụ án.\n" +
                                    "- Mất lái: Chiếm khoảng 15% tổng vụ án.\n" +
                                    "- Không chú ý quan sát: Chiếm khoảng 11% tổng vụ án.\n\n" +
                                    "**Địa điểm trọng điểm:**\n\n" +
                                    "- Thủ Đức: Có số vụ tai nạn cao nhất với 14,24% tổng vụ án.\n" +
                                    "- Quận 7: Chỉ thua Thủ Đức 1 vụ, đặc biệt trên đường Nguyễn Văn Linh.\n" +
                                    "- Đường Phạm Văn Đồng: Cũng là điểm nóng với nhiều vụ va chạm.";
                    break;

                case 'tuoi':
                    displayText.col1 = "\nTừ năm 2022 đến nay, độ tuổi gây tai nạn chủ yếu là từ 25 đến 31 tuổi. Đây là độ tuổi tham gia giao thông nhiều"+
                                        ", nhưng kinh nghiệm lái xe chưa dày dặn. Tai nạn giảm dần ở độ tuổi từ 32 trở lên.\n\n"+
                                        "Năm 2023, nhóm tuổi 25 đến 31 vẫn chiếm đa số vụ tai nạn. Tuy nhiên, số vụ tai nạn ở độ tuổi 53 trở lên đã"+
                                        " giảm mạnh từ 3,91% xuống 1,34%.\n\n"+
                                        "Năm 2022, nhóm tuổi 25 đến 31 gây nhiều tai nạn nhất, tiếp theo là nhóm 18 đến 24. Sau đó, tai"+
                                        " nạn giảm dần ở các độ tuổi lớn hơn.";
                    displayText.col2 = 
                                        "\n**Tình hình tai nạn giao thông theo độ tuổi (Năm 2023):**\n\n" +
                                        "- Nhóm tuổi 18 - 24: Chiếm 7,38% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 25 - 31: Chiếm 37,92% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 32 - 38: Chiếm 29,53% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 39 - 45: Chiếm 19,46% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 46 - 52: Chiếm 4,37% tổng số vụ tai nạn.\n\n" + 
                                        "- Nhóm tuổi 53 trở lên: Chiếm 1,34% tổng số vụ tai nạn."; 
                    displayText.col3 = 
                                        "\n**Tình hình tai nạn giao thông theo độ tuổi (Năm 2023):**\n\n" +
                                        "- Nhóm tuổi 18 - 24: Chiếm 4,63% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 25 - 31: Chiếm 33,81% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 32 - 38: Chiếm 26,33% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 39 - 45: Chiếm 23,13% tổng số vụ tai nạn.\n\n" +
                                        "- Nhóm tuổi 46 - 52: Chiếm 8,19% tổng số vụ tai nạn.\n\n" + 
                                        "- Nhóm tuổi 53 trở lên: Chiếm 3,91% tổng số vụ tai nạn."; 
                    break;
                
                case 'gio':
                    displayText.col1 = "\nTai nạn giao thông thường xảy ra vào buổi chiều và tối, từ 12h đến 24h, đặc biệt là giờ cao điểm."+
                                        " Tai nạn ít hơn vào buổi trưa và buổi sáng.\n\n"+
                                        "Năm 2023, tai nạn vẫn chủ yếu diễn ra vào chiều tối, nhưng buổi trưa có sự tăng nhẹ về số vụ tai nạn."+
                                        " Buổi sáng có giảm nhẹ số lượng tai nạn.\n\n"+
                                        "Năm 2022, tai nạn tập trung vào buổi chiều tối từ 12h đến 24h. Buổi sáng và buổi trưa có số lượng tai nạn ít hơn.";
                    displayText.col2 = 
                                        "\n**Tình hình tai nạn giao thông theo giờ năm 2023:**\n\n" +
                                        "- Buổi sáng (5h - 10h): Chiếm 22,52% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi trưa (10h - 12h): Chiếm 9,04% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi chiều (12h - 18h): Chiếm 32,44% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi tối (18h - 24h): Chiếm 35,49% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi khuya (24h - 5h): Chiếm 0,57% tổng số vụ tai nạn.";
                    displayText.col3 = 
                                        "\n**Tình hình tai nạn giao thông theo giờ năm 2022:**\n\n" +
                                        "- Buổi sáng (5h - 10h): Chiếm 27,44% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi trưa (10h - 12h): Chiếm 5,14% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi chiều (12h - 18h): Chiếm 30,05% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi tối (18h - 24h): Chiếm 37,01% tổng số vụ tai nạn.\n\n" +
                                        "- Buổi khuya (24h - 5h): Chiếm 0,36% tổng số vụ tai nạn.";
                    break;
                    
                default:
                    break;
            }
        }
        updateDisplayText(); // Cập nhật văn bản hiển thị
    });

    // Khởi tạo văn bản hiển thị
    updateDisplayText();
