import random
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Database từ vựng A2-B2 level
VOCABULARY_DATABASE = [
    {
        "word": "achieve",
        "meaning": "đạt được, hoàn thành",
        "example": "She worked hard to achieve her goals.",
    },
    {
        "word": "adventure",
        "meaning": "cuộc phiêu lưu",
        "example": "The book tells the story of a great adventure.",
    },
    {
        "word": "ancient",
        "meaning": "cổ xưa, cổ đại",
        "example": "We visited an ancient temple in Greece.",
    },
    {
        "word": "appreciate",
        "meaning": "đánh giá cao, cảm kích",
        "example": "I really appreciate your help with this project.",
    },
    {
        "word": "approach",
        "meaning": "tiếp cận, phương pháp",
        "example": "We need a new approach to solve this problem.",
    },
    {
        "word": "attitude",
        "meaning": "thái độ",
        "example": "Your positive attitude makes a big difference.",
    },
    {
        "word": "available",
        "meaning": "có sẵn, có thể sử dụng",
        "example": "The new software will be available next month.",
    },
    {
        "word": "benefit",
        "meaning": "lợi ích, hưởng lợi",
        "example": "Regular exercise has many health benefits.",
    },
    {
        "word": "challenge",
        "meaning": "thử thách, thách thức",
        "example": "Learning a new language is a great challenge.",
    },
    {
        "word": "comfortable",
        "meaning": "thoải mái, dễ chịu",
        "example": "This chair is very comfortable to sit in.",
    },
    {
        "word": "community",
        "meaning": "cộng đồng",
        "example": "Our local community is very friendly.",
    },
    {
        "word": "compare",
        "meaning": "so sánh",
        "example": "Let's compare these two options carefully.",
    },
    {
        "word": "consider",
        "meaning": "xem xét, cân nhắc",
        "example": "Please consider all the possibilities.",
    },
    {
        "word": "continue",
        "meaning": "tiếp tục",
        "example": "We will continue our meeting after lunch.",
    },
    {
        "word": "create",
        "meaning": "tạo ra, sáng tạo",
        "example": "Artists create beautiful paintings.",
    },
    {
        "word": "culture",
        "meaning": "văn hóa",
        "example": "I love learning about different cultures.",
    },
    {
        "word": "decision",
        "meaning": "quyết định",
        "example": "Making the right decision is not always easy.",
    },
    {
        "word": "develop",
        "meaning": "phát triển",
        "example": "Children develop quickly in their first years.",
    },
    {
        "word": "different",
        "meaning": "khác nhau",
        "example": "People have different opinions about this topic.",
    },
    {
        "word": "difficult",
        "meaning": "khó khăn",
        "example": "This math problem is very difficult.",
    },
    {
        "word": "education",
        "meaning": "giáo dục",
        "example": "Education is very important for everyone.",
    },
    {
        "word": "environment",
        "meaning": "môi trường",
        "example": "We must protect our environment.",
    },
    {
        "word": "experience",
        "meaning": "kinh nghiệm, trải nghiệm",
        "example": "Traveling gives you valuable experience.",
    },
    {
        "word": "explain",
        "meaning": "giải thích",
        "example": "Can you explain how this works?",
    },
    {
        "word": "familiar",
        "meaning": "quen thuộc",
        "example": "This place looks familiar to me.",
    },
    {
        "word": "famous",
        "meaning": "nổi tiếng",
        "example": "She is a famous singer in our country.",
    },
    {
        "word": "favorite",
        "meaning": "yêu thích",
        "example": "Pizza is my favorite food.",
    },
    {
        "word": "foreign",
        "meaning": "nước ngoài",
        "example": "I love learning foreign languages.",
    },
    {
        "word": "freedom",
        "meaning": "tự do",
        "example": "Freedom is a basic human right.",
    },
    {
        "word": "friendly",
        "meaning": "thân thiện",
        "example": "The people here are very friendly.",
    },
    {
        "word": "future",
        "meaning": "tương lai",
        "example": "I'm excited about the future.",
    },
    {
        "word": "generation",
        "meaning": "thế hệ",
        "example": "Younger generations use technology differently.",
    },
    {
        "word": "government",
        "meaning": "chính phủ",
        "example": "The government announced new policies.",
    },
    {
        "word": "happiness",
        "meaning": "hạnh phúc",
        "example": "Money doesn't always bring happiness.",
    },
    {
        "word": "important",
        "meaning": "quan trọng",
        "example": "It's important to eat healthy food.",
    },
    {
        "word": "improve",
        "meaning": "cải thiện",
        "example": "I want to improve my English skills.",
    },
    {
        "word": "include",
        "meaning": "bao gồm",
        "example": "The price includes breakfast and dinner.",
    },
    {
        "word": "increase",
        "meaning": "tăng lên",
        "example": "The price of gas continues to increase.",
    },
    {
        "word": "information",
        "meaning": "thông tin",
        "example": "I need more information about this topic.",
    },
    {
        "word": "interest",
        "meaning": "quan tâm, sở thích",
        "example": "I have a great interest in music.",
    },
    {
        "word": "international",
        "meaning": "quốc tế",
        "example": "This is an international company.",
    },
    {
        "word": "knowledge",
        "meaning": "kiến thức",
        "example": "Knowledge is power.",
    },
    {
        "word": "language",
        "meaning": "ngôn ngữ",
        "example": "English is a global language.",
    },
    {
        "word": "learn",
        "meaning": "học",
        "example": "Children learn quickly when they're young.",
    },
    {
        "word": "manage",
        "meaning": "quản lý, xoay xở",
        "example": "She manages a team of 20 people.",
    },
    {
        "word": "modern",
        "meaning": "hiện đại",
        "example": "This is a very modern building.",
    },
    {
        "word": "necessary",
        "meaning": "cần thiết",
        "example": "It's necessary to study hard for exams.",
    },
    {
        "word": "opportunity",
        "meaning": "cơ hội",
        "example": "This job is a great opportunity for me.",
    },
    {
        "word": "organization",
        "meaning": "tổ chức",
        "example": "She works for a non-profit organization.",
    },
    {
        "word": "popular",
        "meaning": "phổ biến, nổi tiếng",
        "example": "This song is very popular among teenagers.",
    },
    {
        "word": "possible",
        "meaning": "có thể",
        "example": "Is it possible to finish this today?",
    },
    {
        "word": "prepare",
        "meaning": "chuẩn bị",
        "example": "I need to prepare for my presentation.",
    },
    {
        "word": "present",
        "meaning": "hiện tại, trình bày",
        "example": "At present, I'm studying at university.",
    },
    {
        "word": "problem",
        "meaning": "vấn đề",
        "example": "We need to solve this problem quickly.",
    },
    {
        "word": "process",
        "meaning": "quá trình, xử lý",
        "example": "Learning a language is a long process.",
    },
    {
        "word": "program",
        "meaning": "chương trình",
        "example": "This computer program is very useful.",
    },
    {
        "word": "provide",
        "meaning": "cung cấp",
        "example": "The school provides free lunch for students.",
    },
    {
        "word": "purpose",
        "meaning": "mục đích",
        "example": "What is the purpose of this meeting?",
    },
    {
        "word": "quality",
        "meaning": "chất lượng",
        "example": "This product has excellent quality.",
    },
    {
        "word": "question",
        "meaning": "câu hỏi",
        "example": "Do you have any questions about this?",
    },
    {
        "word": "relationship",
        "meaning": "mối quan hệ",
        "example": "Good relationships are important in life.",
    },
    {
        "word": "remember",
        "meaning": "nhớ",
        "example": "I can't remember his name.",
    },
    {
        "word": "research",
        "meaning": "nghiên cứu",
        "example": "Scientists do research to find new medicines.",
    },
    {
        "word": "resource",
        "meaning": "tài nguyên",
        "example": "Water is a precious resource.",
    },
    {
        "word": "result",
        "meaning": "kết quả",
        "example": "The test results will be ready tomorrow.",
    },
    {
        "word": "science",
        "meaning": "khoa học",
        "example": "Science helps us understand the world.",
    },
    {
        "word": "society",
        "meaning": "xã hội",
        "example": "Technology has changed our society.",
    },
    {
        "word": "special",
        "meaning": "đặc biệt",
        "example": "Today is a special day for me.",
    },
    {
        "word": "success",
        "meaning": "thành công",
        "example": "Hard work leads to success.",
    },
    {
        "word": "support",
        "meaning": "hỗ trợ, ủng hộ",
        "example": "My family always supports my decisions.",
    },
    {
        "word": "system",
        "meaning": "hệ thống",
        "example": "The school has a good computer system.",
    },
    {
        "word": "technology",
        "meaning": "công nghệ",
        "example": "Modern technology makes life easier.",
    },
    {
        "word": "tradition",
        "meaning": "truyền thống",
        "example": "We follow many family traditions.",
    },
    {
        "word": "understand",
        "meaning": "hiểu",
        "example": "I don't understand this question.",
    },
    {
        "word": "university",
        "meaning": "đại học",
        "example": "She studies at a famous university.",
    },
    {
        "word": "usually",
        "meaning": "thường xuyên",
        "example": "I usually wake up at 7 AM.",
    },
    {
        "word": "various",
        "meaning": "nhiều loại khác nhau",
        "example": "The store sells various types of books.",
    },
    {
        "word": "wonderful",
        "meaning": "tuyệt vời",
        "example": "We had a wonderful time at the beach.",
    },
    {
        "word": "achieve",
        "meaning": "đạt được",
        "example": "She worked hard to achieve her dream of becoming a doctor.",
    },
    {
        "word": "adapt",
        "meaning": "thích nghi",
        "example": "He quickly adapted to the new work environment.",
    },
    {
        "word": "ambitious",
        "meaning": "tham vọng",
        "example": "Her ambitious goals inspired her team to work harder.",
    },
    {
        "word": "analyze",
        "meaning": "phân tích",
        "example": "The scientist analyzed the data to find patterns.",
    },
    {
        "word": "anxious",
        "meaning": "lo lắng",
        "example": "She felt anxious before her final exam.",
    },
    {
        "word": "assume",
        "meaning": "cho rằng",
        "example": "I assumed he was coming to the party, but he didn’t show up.",
    },
    {
        "word": "benefit",
        "meaning": "lợi ích",
        "example": "Regular exercise has many health benefits.",
    },
    {
        "word": "challenge",
        "meaning": "thử thách",
        "example": "Learning a new language is a rewarding challenge.",
    },
    {
        "word": "commitment",
        "meaning": "cam kết",
        "example": "His commitment to the project impressed the team.",
    },
    {
        "word": "complex",
        "meaning": "phức tạp",
        "example": "The instructions were too complex to follow.",
    },
    {
        "word": "concentrate",
        "meaning": "tập trung",
        "example": "It’s hard to concentrate with so much noise around.",
    },
    {
        "word": "confident",
        "meaning": "tự tin",
        "example": "She felt confident about her presentation skills.",
    },
    {
        "word": "considerable",
        "meaning": "đáng kể",
        "example": "The project required a considerable amount of time.",
    },
    {
        "word": "contribute",
        "meaning": "đóng góp",
        "example": "Everyone should contribute to protecting the environment.",
    },
    {
        "word": "curious",
        "meaning": "tò mò",
        "example": "The children were curious about the new science experiment.",
    },
    {
        "word": "debate",
        "meaning": "tranh luận",
        "example": "The students had a lively debate about climate change.",
    },
    {
        "word": "dedicated",
        "meaning": "cống hiến",
        "example": "She is a dedicated teacher who cares about her students.",
    },
    {
        "word": "demand",
        "meaning": "yêu cầu",
        "example": "The job demands strong communication skills.",
    },
    {
        "word": "diverse",
        "meaning": "đa dạng",
        "example": "The city has a diverse population from many countries.",
    },
    {
        "word": "efficient",
        "meaning": "hiệu quả",
        "example": "The new system is more efficient than the old one.",
    },
    {
        "word": "encourage",
        "meaning": "khuyến khích",
        "example": "Teachers should encourage students to ask questions.",
    },
    {
        "word": "establish",
        "meaning": "thiết lập",
        "example": "They established a new company in 2020.",
    },
    {
        "word": "expand",
        "meaning": "mở rộng",
        "example": "The business plans to expand into international markets.",
    },
    {
        "word": "flexible",
        "meaning": "linh hoạt",
        "example": "Her flexible schedule allows her to travel often.",
    },
    {
        "word": "generate",
        "meaning": "tạo ra",
        "example": "The campaign generated a lot of public interest.",
    },
    {
        "word": "gradually",
        "meaning": "dần dần",
        "example": "Her health improved gradually after the surgery.",
    },
    {
        "word": "impact",
        "meaning": "tác động",
        "example": "Social media has a significant impact on communication.",
    },
    {
        "word": "implement",
        "meaning": "thực hiện",
        "example": "The company implemented new safety measures.",
    },
    {
        "word": "inspire",
        "meaning": "truyền cảm hứng",
        "example": "Her speech inspired many people to take action.",
    },
    {
        "word": "invest",
        "meaning": "đầu tư",
        "example": "He decided to invest in a new startup company.",
    },
    {
        "word": "logical",
        "meaning": "hợp lý",
        "example": "Her argument was clear and logical.",
    },
    {
        "word": "maintain",
        "meaning": "duy trì",
        "example": "It’s important to maintain a healthy lifestyle.",
    },
    {
        "word": "motivate",
        "meaning": "thúc đẩy",
        "example": "The coach motivated the team to perform better.",
    },
    {
        "word": "negotiate",
        "meaning": "đàm phán",
        "example": "They negotiated a better deal with the supplier.",
    },
    {
        "word": "objective",
        "meaning": "mục tiêu",
        "example": "Her main objective is to improve her English skills.",
    },
    {
        "word": "opportunity",
        "meaning": "cơ hội",
        "example": "Studying abroad is a great opportunity to learn new cultures.",
    },
    {
        "word": "overcome",
        "meaning": "vượt qua",
        "example": "She overcame many challenges to succeed in her career.",
    },
    {
        "word": "perspective",
        "meaning": "góc nhìn",
        "example": "Traveling gave him a new perspective on life.",
    },
    {
        "word": "precise",
        "meaning": "chính xác",
        "example": "The instructions must be precise to avoid mistakes.",
    },
    {
        "word": "priority",
        "meaning": "ưu tiên",
        "example": "Safety is the top priority in this factory.",
    },
    {
        "word": "promote",
        "meaning": "thúc đẩy",
        "example": "The campaign promotes awareness of recycling.",
    },
    {
        "word": "reliable",
        "meaning": "đáng tin cậy",
        "example": "This car brand is known for being reliable.",
    },
    {
        "word": "resolve",
        "meaning": "giải quyết",
        "example": "They resolved the conflict through discussion.",
    },
    {
        "word": "significant",
        "meaning": "quan trọng",
        "example": "The discovery was a significant breakthrough in science.",
    },
    {
        "word": "strategy",
        "meaning": "chiến lược",
        "example": "The company developed a new marketing strategy.",
    },
    {
        "word": "succeed",
        "meaning": "thành công",
        "example": "With hard work, she succeeded in her exams.",
    },
    {
        "word": "sustain",
        "meaning": "duy trì",
        "example": "We need to sustain our efforts to protect the environment.",
    },
    {
        "word": "trend",
        "meaning": "xu hướng",
        "example": "Minimalism is a popular trend in design today.",
    },
    {
        "word": "unique",
        "meaning": "độc đáo",
        "example": "Her painting style is truly unique.",
    },
    {
        "word": "voluntary",
        "meaning": "tự nguyện",
        "example": "She does voluntary work at the local charity.",
    },
    {
        "word": "abundant",
        "meaning": "dồi dào",
        "example": "The region has abundant natural resources.",
    },
    {
        "word": "accomplish",
        "meaning": "hoàn thành",
        "example": "She accomplished her goal of running a marathon.",
    },
    {
        "word": "accurate",
        "meaning": "chính xác",
        "example": "His predictions about the weather were accurate.",
    },
    {
        "word": "acknowledge",
        "meaning": "thừa nhận",
        "example": "He acknowledged his mistake during the meeting.",
    },
    {
        "word": "acquire",
        "meaning": "thu được",
        "example": "She acquired new skills through online courses.",
    },
    {
        "word": "adequate",
        "meaning": "đầy đủ",
        "example": "The room was adequate for our needs.",
    },
    {
        "word": "adjust",
        "meaning": "điều chỉnh",
        "example": "He adjusted the settings on his computer.",
    },
    {
        "word": "admire",
        "meaning": "ngưỡng mộ",
        "example": "I admire her dedication to charity work.",
    },
    {
        "word": "advocate",
        "meaning": "ủng hộ",
        "example": "She advocates for equal rights in education.",
    },
    {
        "word": "affect",
        "meaning": "ảnh hưởng",
        "example": "The new law will affect small businesses.",
    },
    {
        "word": "ambiguity",
        "meaning": "sự mơ hồ",
        "example": "The ambiguity of his response confused everyone.",
    },
    {
        "word": "anticipate",
        "meaning": "dự đoán",
        "example": "We anticipate a rise in sales next month.",
    },
    {
        "word": "apparent",
        "meaning": "rõ ràng",
        "example": "Her disappointment was apparent to everyone.",
    },
    {
        "word": "apply",
        "meaning": "áp dụng",
        "example": "You can apply these techniques to improve your writing.",
    },
    {
        "word": "appreciate",
        "meaning": "trân trọng",
        "example": "I appreciate your help with the project.",
    },
    {
        "word": "approach",
        "meaning": "cách tiếp cận",
        "example": "Her approach to problem-solving is very creative.",
    },
    {
        "word": "appropriate",
        "meaning": "phù hợp",
        "example": "This dress is not appropriate for a formal event.",
    },
    {
        "word": "arise",
        "meaning": "nảy sinh",
        "example": "Problems may arise if we don’t plan carefully.",
    },
    {
        "word": "assert",
        "meaning": "khẳng định",
        "example": "He asserted his authority during the discussion.",
    },
    {
        "word": "assess",
        "meaning": "đánh giá",
        "example": "Teachers assess students’ progress regularly.",
    },
    {
        "word": "assist",
        "meaning": "hỗ trợ",
        "example": "She assisted her colleague with the presentation.",
    },
    {
        "word": "attain",
        "meaning": "đạt được",
        "example": "He attained a high score on the exam.",
    },
    {
        "word": "attitude",
        "meaning": "thái độ",
        "example": "Her positive attitude makes her a great leader.",
    },
    {
        "word": "attribute",
        "meaning": "quy cho",
        "example": "She attributed her success to hard work.",
    },
    {
        "word": "authentic",
        "meaning": "chính thống",
        "example": "The restaurant serves authentic Vietnamese phở.",
    },
    {
        "word": "authority",
        "meaning": "quyền lực",
        "example": "The manager has the authority to make decisions.",
    },
    {
        "word": "aware",
        "meaning": "nhận thức",
        "example": "Are you aware of the new company policies?",
    },
    {
        "word": "bias",
        "meaning": "thiên vị",
        "example": "The article showed a clear bias toward one candidate.",
    },
    {
        "word": "capable",
        "meaning": "có khả năng",
        "example": "She is capable of solving complex problems.",
    },
    {
        "word": "clarify",
        "meaning": "làm rõ",
        "example": "Can you clarify what you mean by that statement?",
    },
    {
        "word": "coherent",
        "meaning": "mạch lạc",
        "example": "Her essay was well-written and coherent.",
    },
    {
        "word": "collaborate",
        "meaning": "hợp tác",
        "example": "The teams collaborated on a new software project.",
    },
    {
        "word": "commence",
        "meaning": "bắt đầu",
        "example": "The meeting will commence at 9 a.m.",
    },
    {
        "word": "compensate",
        "meaning": "bù đắp",
        "example": "The company compensated employees for overtime work.",
    },
    {
        "word": "comprehensive",
        "meaning": "toàn diện",
        "example": "The report provides a comprehensive overview of the issue.",
    },
    {
        "word": "comprise",
        "meaning": "bao gồm",
        "example": "The committee comprises members from different departments.",
    },
    {
        "word": "compulsory",
        "meaning": "bắt buộc",
        "example": "Wearing a helmet is compulsory for motorbike riders.",
    },
    {
        "word": "conclude",
        "meaning": "kết luận",
        "example": "The study concluded that exercise improves mental health.",
    },
    {
        "word": "conduct",
        "meaning": "thực hiện",
        "example": "They conducted a survey to gather opinions.",
    },
    {
        "word": "consequence",
        "meaning": "hậu quả",
        "example": "Ignoring the problem could have serious consequences.",
    },
    {
        "word": "consistent",
        "meaning": "nhất quán",
        "example": "Her performance has been consistent throughout the year.",
    },
    {
        "word": "constraint",
        "meaning": "ràng buộc",
        "example": "Time constraints prevented us from finishing the project.",
    },
    {
        "word": "consult",
        "meaning": "tham khảo",
        "example": "You should consult a doctor if the pain persists.",
    },
    {
        "word": "consume",
        "meaning": "tiêu thụ",
        "example": "This car consumes less fuel than older models.",
    },
    {
        "word": "context",
        "meaning": "bối cảnh",
        "example": "His words were misunderstood because of the context.",
    },
    {
        "word": "contradict",
        "meaning": "mâu thuẫn",
        "example": "His statement contradicts the facts we have.",
    },
    {
        "word": "controversial",
        "meaning": "gây tranh cãi",
        "example": "The new policy is highly controversial among employees.",
    },
    {
        "word": "convey",
        "meaning": "truyền đạt",
        "example": "Her speech conveyed a powerful message.",
    },
    {
        "word": "cooperate",
        "meaning": "hợp tác",
        "example": "The two companies cooperated on a joint project.",
    },
    {
        "word": "cope",
        "meaning": "đối phó",
        "example": "She learned to cope with stress through meditation.",
    },
    {
        "word": "crucial",
        "meaning": "quan trọng",
        "example": "His support was crucial to the project’s success.",
    },
    {
        "word": "decline",
        "meaning": "suy giảm",
        "example": "The quality of service has declined recently.",
    },
    {
        "word": "demonstrate",
        "meaning": "thể hiện",
        "example": "She demonstrated her skills during the interview.",
    },
    {
        "word": "derive",
        "meaning": "bắt nguồn",
        "example": "Many English words derive from Latin.",
    },
    {
        "word": "deserve",
        "meaning": "xứng đáng",
        "example": "You deserve a break after all your hard work.",
    },
    {
        "word": "detect",
        "meaning": "phát hiện",
        "example": "The device can detect changes in temperature.",
    },
    {
        "word": "devise",
        "meaning": "thiết kế",
        "example": "They devised a plan to improve productivity.",
    },
    {
        "word": "diminish",
        "meaning": "giảm bớt",
        "example": "The pain diminished after taking the medicine.",
    },
    {
        "word": "discipline",
        "meaning": "kỷ luật",
        "example": "Maintaining discipline is key to success.",
    },
    {
        "word": "discourage",
        "meaning": "làm nản lòng",
        "example": "Don’t let failure discourage you from trying again.",
    },
    {
        "word": "dispute",
        "meaning": "tranh chấp",
        "example": "There was a dispute over the contract terms.",
    },
    {
        "word": "distinguish",
        "meaning": "phân biệt",
        "example": "It’s hard to distinguish between the two brands.",
    },
    {
        "word": "divert",
        "meaning": "chuyển hướng",
        "example": "They diverted the funds to a new project.",
    },
    {
        "word": "dominant",
        "meaning": "thống trị",
        "example": "The company is the dominant player in the market.",
    },
    {
        "word": "durable",
        "meaning": "bền vững",
        "example": "These shoes are durable and last for years.",
    },
    {
        "word": "dynamic",
        "meaning": "năng động",
        "example": "The team has a dynamic approach to innovation.",
    },
    {
        "word": "eliminate",
        "meaning": "loại bỏ",
        "example": "We need to eliminate errors in the report.",
    },
    {
        "word": "emerge",
        "meaning": "xuất hiện",
        "example": "New opportunities emerged after the meeting.",
    },
    {
        "word": "emphasize",
        "meaning": "nhấn mạnh",
        "example": "The teacher emphasized the importance of homework.",
    },
    {
        "word": "enable",
        "meaning": "cho phép",
        "example": "This software enables us to work faster.",
    },
    {
        "word": "encounter",
        "meaning": "gặp phải",
        "example": "She encountered some difficulties during the project.",
    },
    {
        "word": "endorse",
        "meaning": "tán thành",
        "example": "The committee endorsed the new proposal.",
    },
    {
        "word": "engage",
        "meaning": "tham gia",
        "example": "Students are encouraged to engage in class discussions.",
    },
    {
        "word": "enhance",
        "meaning": "nâng cao",
        "example": "Training will enhance your professional skills.",
    },
    {
        "word": "ensure",
        "meaning": "bảo đảm",
        "example": "Please ensure all doors are locked before leaving.",
    },
    {
        "word": "essential",
        "meaning": "thiết yếu",
        "example": "Water is essential for survival.",
    },
    {
        "word": "evaluate",
        "meaning": "đánh giá",
        "example": "The team will evaluate the project’s success next week.",
    },
    {
        "word": "evident",
        "meaning": "hiển nhiên",
        "example": "His talent was evident from a young age.",
    },
    {
        "word": "evolve",
        "meaning": "tiến hóa",
        "example": "The company evolved into a global brand.",
    },
    {
        "word": "exceed",
        "meaning": "vượt quá",
        "example": "The results exceeded our expectations.",
    },
    {
        "word": "exhibit",
        "meaning": "triển lãm",
        "example": "The museum exhibits ancient artifacts.",
    },
    {
        "word": "expertise",
        "meaning": "chuyên môn",
        "example": "Her expertise in coding is impressive.",
    },
    {
        "word": "exploit",
        "meaning": "khai thác",
        "example": "They exploited the opportunity to expand their business.",
    },
    {
        "word": "expose",
        "meaning": "phơi bày",
        "example": "The report exposed corruption in the organization.",
    },
    {
        "word": "facilitate",
        "meaning": "thúc đẩy",
        "example": "The new system facilitates faster communication.",
    },
    {
        "word": "feasible",
        "meaning": "khả thi",
        "example": "The plan seems feasible with our current resources.",
    },
    {
        "word": "flourish",
        "meaning": "phát triển mạnh",
        "example": "The business flourished after the new campaign.",
    },
    {
        "word": "formulate",
        "meaning": "xây dựng",
        "example": "They formulated a strategy to increase sales.",
    },
    {
        "word": "foster",
        "meaning": "thúc đẩy",
        "example": "The program fosters creativity in young students.",
    },
    {
        "word": "fundamental",
        "meaning": "cơ bản",
        "example": "Learning grammar is fundamental to language skills.",
    },
    {
        "word": "generate",
        "meaning": "tạo ra",
        "example": "The campaign generated a lot of interest.",
    },
    {
        "word": "genuine",
        "meaning": "chân thật",
        "example": "Her smile was genuine and welcoming.",
    },
    {
        "word": "grasp",
        "meaning": "nắm bắt",
        "example": "He quickly grasped the main ideas of the lecture.",
    },
    {
        "word": "guarantee",
        "meaning": "bảo đảm",
        "example": "This product comes with a two-year guarantee.",
    },
    {
        "word": "hypothesis",
        "meaning": "giả thuyết",
        "example": "The scientist tested her hypothesis with experiments.",
    },
    {
        "word": "identify",
        "meaning": "xác định",
        "example": "Can you identify the source of the problem?",
    },
    {
        "word": "illustrate",
        "meaning": "minh họa",
        "example": "The teacher used examples to illustrate the concept.",
    },
    {
        "word": "impartial",
        "meaning": "khách quan",
        "example": "Judges must remain impartial in court.",
    },
    {
        "word": "imply",
        "meaning": "ám chỉ",
        "example": "Her silence implied that she was upset.",
    },
    {
        "word": "incentive",
        "meaning": "động lực",
        "example": "Bonuses are an incentive for employees to work harder.",
    },
    {
        "word": "incorporate",
        "meaning": "kết hợp",
        "example": "The design incorporates modern and traditional elements.",
    },
    {
        "word": "indicate",
        "meaning": "chỉ ra",
        "example": "The data indicates a rise in unemployment.",
    },
    {
        "word": "inevitable",
        "meaning": "không thể tránh khỏi",
        "example": "Change is inevitable in a fast-moving world.",
    },
    {
        "word": "infer",
        "meaning": "suy ra",
        "example": "From her tone, I inferred she was unhappy.",
    },
    {
        "word": "inhibit",
        "meaning": "ngăn cản",
        "example": "Fear can inhibit people from taking risks.",
    },
    {
        "word": "initiate",
        "meaning": "khởi xướng",
        "example": "She initiated a program to help local communities.",
    },
    {
        "word": "innovative",
        "meaning": "đổi mới",
        "example": "The company is known for its innovative products.",
    },
    {
        "word": "insight",
        "meaning": "sự hiểu biết",
        "example": "Her book provides deep insight into human behavior.",
    },
    {
        "word": "integrate",
        "meaning": "hòa nhập",
        "example": "New employees need time to integrate into the team.",
    },
    {
        "word": "intense",
        "meaning": "mãnh liệt",
        "example": "The competition was intense this year.",
    },
    {
        "word": "interfere",
        "meaning": "can thiệp",
        "example": "Don’t interfere in matters that don’t concern you.",
    },
    {
        "word": "interpret",
        "meaning": "giải thích",
        "example": "She interpreted the poem in a unique way.",
    },
    {
        "word": "justify",
        "meaning": "bào chữa",
        "example": "He tried to justify his absence from the meeting.",
    },
    {
        "word": "legitimate",
        "meaning": "hợp pháp",
        "example": "They raised a legitimate concern about safety.",
    },
    {
        "word": "likelihood",
        "meaning": "khả năng",
        "example": "There’s a high likelihood of rain this afternoon.",
    },
    {
        "word": "manipulate",
        "meaning": "thao túng",
        "example": "The software can manipulate large datasets easily.",
    },
    {
        "word": "maximize",
        "meaning": "tối đa hóa",
        "example": "We need to maximize our resources to finish on time.",
    },
    {
        "word": "moderate",
        "meaning": "vừa phải",
        "example": "The price increase was moderate and affordable.",
    },
    {
        "word": "monitor",
        "meaning": "giám sát",
        "example": "They monitor the progress of the project weekly.",
    },
    {
        "word": "mutual",
        "meaning": "lẫn nhau",
        "example": "We have a mutual interest in environmental issues.",
    },
    {
        "word": "notion",
        "meaning": "khái niệm",
        "example": "The notion of freedom varies across cultures.",
    },
    {
        "word": "objective",
        "meaning": "khách quan",
        "example": "The journalist tried to remain objective in her reporting.",
    },
    {
        "word": "obliged",
        "meaning": "bắt buộc",
        "example": "We are obliged to follow the safety regulations.",
    },
    {
        "word": "observe",
        "meaning": "quan sát",
        "example": "She observed the behavior of the animals in the wild.",
    },
    {
        "word": "obstacle",
        "meaning": "trở ngại",
        "example": "Lack of funding was a major obstacle to the project.",
    },
    {
        "word": "occasion",
        "meaning": "dịp",
        "example": "The party was held on the occasion of her birthday.",
    },
    {
        "word": "ongoing",
        "meaning": "đang diễn ra",
        "example": "The research is an ongoing process.",
    },
    {
        "word": "optimistic",
        "meaning": "lạc quan",
        "example": "She remains optimistic about her future career.",
    },
    {
        "word": "outline",
        "meaning": "phác thảo",
        "example": "He outlined his plan for the new project.",
    },
    {
        "word": "overlook",
        "meaning": "bỏ qua",
        "example": "Don’t overlook the importance of small details.",
    },
    {
        "word": "participate",
        "meaning": "tham gia",
        "example": "Everyone is encouraged to participate in the event.",
    },
    {
        "word": "perceive",
        "meaning": "nhận thức",
        "example": "People perceive the issue differently based on their experiences.",
    },
    {
        "word": "permanent",
        "meaning": "vĩnh viễn",
        "example": "The changes to the law are permanent.",
    },
    {
        "word": "persist",
        "meaning": "kiên trì",
        "example": "If the problem persists, please contact support.",
    },
    {
        "word": "persuade",
        "meaning": "thuyết phục",
        "example": "She persuaded her team to adopt the new plan.",
    },
    {
        "word": "pessimistic",
        "meaning": "bi quan",
        "example": "He’s pessimistic about the economy’s recovery.",
    },
    {
        "word": "plausible",
        "meaning": "hợp lý",
        "example": "Her explanation seemed plausible at first.",
    },
    {
        "word": "precede",
        "meaning": "đi trước",
        "example": "A short introduction preceded the main event.",
    },
    {
        "word": "predict",
        "meaning": "dự đoán",
        "example": "It’s hard to predict the outcome of the election.",
    },
    {
        "word": "preserve",
        "meaning": "bảo tồn",
        "example": "We must preserve our cultural heritage.",
    },
    {
        "word": "prevalent",
        "meaning": "phổ biến",
        "example": "Smartphones are prevalent in modern society.",
    },
    {
        "word": "proceed",
        "meaning": "tiếp tục",
        "example": "Let’s proceed with the next item on the agenda.",
    },
    {
        "word": "profound",
        "meaning": "sâu sắc",
        "example": "His speech had a profound impact on the audience.",
    },
    {
        "word": "prohibit",
        "meaning": "cấm",
        "example": "Smoking is prohibited in public areas.",
    },
    {
        "word": "prospect",
        "meaning": "triển vọng",
        "example": "The job offers good prospects for career growth.",
    },
    {
        "word": "pursue",
        "meaning": "theo đuổi",
        "example": "She decided to pursue a degree in engineering.",
    },
    {
        "word": "rational",
        "meaning": "hợp lý",
        "example": "His decision was based on rational thinking.",
    },
    {
        "word": "reinforce",
        "meaning": "tăng cường",
        "example": "The evidence reinforced her argument.",
    },
    {
        "word": "reject",
        "meaning": "từ chối",
        "example": "They rejected the proposal due to budget issues.",
    },
    {
        "word": "relevant",
        "meaning": "liên quan",
        "example": "Please provide relevant information for the report.",
    },
    {
        "word": "reluctant",
        "meaning": "miễn cưỡng",
        "example": "He was reluctant to share his personal details.",
    },
    {
        "word": "rely",
        "meaning": "dựa vào",
        "example": "You can rely on her to finish the task on time.",
    },
    {
        "word": "remarkable",
        "meaning": "đáng chú ý",
        "example": "Her progress in learning English is remarkable.",
    },
    {
        "word": "resemble",
        "meaning": "giống",
        "example": "The painting resembles a famous artwork.",
    },
    {
        "word": "restrict",
        "meaning": "hạn chế",
        "example": "New laws restrict the use of plastic bags.",
    },
    {
        "word": "retain",
        "meaning": "giữ lại",
        "example": "The company aims to retain its best employees.",
    },
    {
        "word": "reveal",
        "meaning": "tiết lộ",
        "example": "The study revealed surprising results.",
    },
    {
        "word": "rigid",
        "meaning": "cứng nhắc",
        "example": "The rules are too rigid and need to be revised.",
    },
    {
        "word": "scope",
        "meaning": "phạm vi",
        "example": "The project is beyond the scope of our budget.",
    },
    {
        "word": "secure",
        "meaning": "bảo vệ",
        "example": "The system is designed to secure sensitive data.",
    },
    {
        "word": "seek",
        "meaning": "tìm kiếm",
        "example": "She is seeking advice on career planning.",
    },
    {
        "word": "selective",
        "meaning": "lựa chọn",
        "example": "The school is very selective about its students.",
    },
    {
        "word": "sensitive",
        "meaning": "nhạy cảm",
        "example": "He is very sensitive to criticism.",
    },
    {
        "word": "shift",
        "meaning": "chuyển đổi",
        "example": "There has been a shift in public opinion.",
    },
    {
        "word": "simulate",
        "meaning": "mô phỏng",
        "example": "The software can simulate real-world scenarios.",
    },
    {
        "word": "sole",
        "meaning": "duy nhất",
        "example": "She was the sole survivor of the accident.",
    },
    {
        "word": "specify",
        "meaning": "chỉ định",
        "example": "Please specify your requirements for the project.",
    },
    {
        "word": "speculate",
        "meaning": "suy đoán",
        "example": "People speculated about the cause of the delay.",
    },
    {
        "word": "spontaneous",
        "meaning": "tự nhiên",
        "example": "Her spontaneous laughter brightened the room.",
    },
    {
        "word": "stabilize",
        "meaning": "ổn định",
        "example": "The economy began to stabilize after the crisis.",
    },
    {
        "word": "stimulate",
        "meaning": "kích thích",
        "example": "The program aims to stimulate economic growth.",
    },
    {
        "word": "submit",
        "meaning": "nộp",
        "example": "Please submit your application by Friday.",
    },
    {
        "word": "subsequent",
        "meaning": "tiếp theo",
        "example": "Subsequent events proved her theory correct.",
    },
    {
        "word": "substantial",
        "meaning": "đáng kể",
        "example": "The company made a substantial profit this year.",
    },
    {
        "word": "subtle",
        "meaning": "tinh tế",
        "example": "The changes to the design were subtle but effective.",
    },
    {
        "word": "sufficient",
        "meaning": "đủ",
        "example": "We have sufficient funds to complete the project.",
    },
    {
        "word": "superior",
        "meaning": "ưu việt",
        "example": "Their product is superior in quality and price.",
    },
    {
        "word": "supplement",
        "meaning": "bổ sung",
        "example": "She takes vitamins to supplement her diet.",
    },
    {
        "word": "surpass",
        "meaning": "vượt qua",
        "example": "Her performance surpassed all expectations.",
    },
    {
        "word": "susceptible",
        "meaning": "dễ bị ảnh hưởng",
        "example": "Young children are susceptible to colds.",
    },
    {
        "word": "temporary",
        "meaning": "tạm thời",
        "example": "The road closure is only temporary.",
    },
    {
        "word": "tendency",
        "meaning": "xu hướng",
        "example": "He has a tendency to procrastinate.",
    },
    {
        "word": "thrive",
        "meaning": "phát triển",
        "example": "Small businesses thrive in this vibrant city.",
    },
    {
        "word": "tolerate",
        "meaning": "chấp nhận",
        "example": "She cannot tolerate loud noises.",
    },
    {
        "word": "transform",
        "meaning": "chuyển đổi",
        "example": "The renovation transformed the old building.",
    },
    {
        "word": "transition",
        "meaning": "sự chuyển đổi",
        "example": "The transition to renewable energy is underway.",
    },
    {
        "word": "transparent",
        "meaning": "minh bạch",
        "example": "The company’s policies are transparent to employees.",
    },
    {
        "word": "trigger",
        "meaning": "kích hoạt",
        "example": "The news triggered a wave of protests.",
    },
    {
        "word": "ultimate",
        "meaning": "cuối cùng",
        "example": "Her ultimate goal is to start her own company.",
    },
    {
        "word": "undergo",
        "meaning": "trải qua",
        "example": "The city will undergo major development next year.",
    },
    {
        "word": "undertake",
        "meaning": "đảm nhận",
        "example": "She undertook a challenging research project.",
    },
    {
        "word": "unify",
        "meaning": "thống nhất",
        "example": "The agreement aims to unify the two organizations.",
    },
    {
        "word": "utilize",
        "meaning": "sử dụng",
        "example": "We should utilize renewable energy sources.",
    },
    {
        "word": "valid",
        "meaning": "hợp lệ",
        "example": "You need a valid passport to travel abroad.",
    },
    {
        "word": "verify",
        "meaning": "xác minh",
        "example": "Please verify your identity before logging in.",
    },
    {
        "word": "versatile",
        "meaning": "đa năng",
        "example": "This tool is versatile and can be used for many tasks.",
    },
    {
        "word": "viable",
        "meaning": "khả thi",
        "example": "The plan is viable with some adjustments.",
    },
    {
        "word": "vigorous",
        "meaning": "mạnh mẽ",
        "example": "The team made a vigorous effort to meet the deadline.",
    },
    {
        "word": "vision",
        "meaning": "tầm nhìn",
        "example": "The leader shared her vision for the company’s future.",
    },
    {
        "word": "vulnerable",
        "meaning": "dễ bị tổn thương",
        "example": "The elderly are vulnerable to extreme weather.",
    },
    {
        "word": "widespread",
        "meaning": "phổ biến",
        "example": "The disease has become widespread in the region.",
    },
    {
        "word": "abandon",
        "meaning": "từ bỏ",
        "example": "They had to abandon the project due to lack of funding.",
    },
    {
        "word": "abrupt",
        "meaning": "đột ngột",
        "example": "The meeting ended with an abrupt decision.",
    },
    {
        "word": "absorb",
        "meaning": "hấp thụ",
        "example": "Sponges absorb water quickly.",
    },
    {
        "word": "accelerate",
        "meaning": "tăng tốc",
        "example": "The company aims to accelerate its growth this year.",
    },
    {
        "word": "accessible",
        "meaning": "dễ tiếp cận",
        "example": "The website is accessible to users with disabilities.",
    },
    {
        "word": "accommodate",
        "meaning": "đáp ứng",
        "example": "The hotel can accommodate up to 200 guests.",
    },
    {
        "word": "accumulate",
        "meaning": "tích lũy",
        "example": "She accumulated a lot of experience working abroad.",
    },
    {
        "word": "adaptable",
        "meaning": "thích nghi được",
        "example": "He is highly adaptable to new situations.",
    },
    {
        "word": "address",
        "meaning": "giải quyết",
        "example": "We need to address the issue of pollution urgently.",
    },
    {
        "word": "adverse",
        "meaning": "bất lợi",
        "example": "The adverse weather delayed the flight.",
    },
    {
        "word": "advocate",
        "meaning": "người ủng hộ",
        "example": "She is an advocate for women’s rights.",
    },
    {
        "word": "affordable",
        "meaning": "phải chăng",
        "example": "The new housing project offers affordable homes.",
    },
    {
        "word": "aggressive",
        "meaning": "hung hăng",
        "example": "His aggressive behavior upset his colleagues.",
    },
    {
        "word": "allocate",
        "meaning": "phân bổ",
        "example": "The budget was allocated to various departments.",
    },
    {
        "word": "ambience",
        "meaning": "bầu không khí",
        "example": "The restaurant has a cozy ambience.",
    },
    {
        "word": "ample",
        "meaning": "dồi dào",
        "example": "There is ample time to finish the assignment.",
    },
    {
        "word": "anticipation",
        "meaning": "sự mong đợi",
        "example": "The anticipation of the results made her nervous.",
    },
    {
        "word": "apprehensive",
        "meaning": "lo lắng",
        "example": "She felt apprehensive about speaking in public.",
    },
    {
        "word": "arbitrary",
        "meaning": "tùy tiện",
        "example": "The decision seemed arbitrary and unfair.",
    },
    {
        "word": "aspire",
        "meaning": "khao khát",
        "example": "She aspires to become a successful writer.",
    },
    {
        "word": "assertive",
        "meaning": "quả quyết",
        "example": "Being assertive helped her negotiate a better deal.",
    },
    {
        "word": "astonishing",
        "meaning": "kinh ngạc",
        "example": "The view from the mountain was astonishing.",
    },
    {
        "word": "attentive",
        "meaning": "chú ý",
        "example": "The teacher was attentive to her students’ needs.",
    },
    {
        "word": "authentic",
        "meaning": "chân thực",
        "example": "The museum displays authentic historical artifacts.",
    },
    {
        "word": "autonomous",
        "meaning": "tự chủ",
        "example": "The new software allows for autonomous operation.",
    },
    {
        "word": "beneficial",
        "meaning": "có lợi",
        "example": "Regular exercise is beneficial to your health.",
    },
    {
        "word": "blend",
        "meaning": "pha trộn",
        "example": "The chef blended spices to create a unique flavor.",
    },
    {
        "word": "bold",
        "meaning": "táo bạo",
        "example": "Her bold decision to start a business paid off.",
    },
    {
        "word": "boost",
        "meaning": "tăng cường",
        "example": "The campaign gave a boost to the company’s sales.",
    },
    {
        "word": "breach",
        "meaning": "vi phạm",
        "example": "The hacker caused a breach in the system’s security.",
    },
    {
        "word": "broaden",
        "meaning": "mở rộng",
        "example": "Traveling can broaden your understanding of cultures.",
    },
    {
        "word": "capture",
        "meaning": "nắm bắt",
        "example": "The photo captured the beauty of the sunset.",
    },
    {
        "word": "cease",
        "meaning": "ngừng",
        "example": "The factory ceased production last month.",
    },
    {
        "word": "cohesive",
        "meaning": "gắn kết",
        "example": "The team’s cohesive efforts led to success.",
    },
    {
        "word": "coincide",
        "meaning": "trùng hợp",
        "example": "Her visit coincided with the festival.",
    },
    {
        "word": "compatible",
        "meaning": "tương thích",
        "example": "This software is compatible with most devices.",
    },
    {
        "word": "compel",
        "meaning": "ép buộc",
        "example": "The evidence compelled him to tell the truth.",
    },
    {
        "word": "complement",
        "meaning": "bổ sung",
        "example": "Her skills complement the team’s strengths.",
    },
    {
        "word": "comply",
        "meaning": "tuân thủ",
        "example": "All employees must comply with safety regulations.",
    },
    {
        "word": "conceive",
        "meaning": "hình dung",
        "example": "It’s hard to conceive how the plan will work.",
    },
    {
        "word": "condense",
        "meaning": "cô đọng",
        "example": "She condensed her speech to fit the time limit.",
    },
    {
        "word": "confine",
        "meaning": "giới hạn",
        "example": "The discussion was confined to budget issues.",
    },
    {
        "word": "confront",
        "meaning": "đối mặt",
        "example": "He confronted his fears and spoke confidently.",
    },
    {
        "word": "consecutive",
        "meaning": "liên tiếp",
        "example": "She won the award for three consecutive years.",
    },
    {
        "word": "consensus",
        "meaning": "sự đồng thuận",
        "example": "The team reached a consensus on the new plan.",
    },
    {
        "word": "constitute",
        "meaning": "tạo thành",
        "example": "These documents constitute legal evidence.",
    },
    {
        "word": "constrain",
        "meaning": "hạn chế",
        "example": "Budget cuts constrained our ability to expand.",
    },
    {
        "word": "contemplate",
        "meaning": "suy ngẫm",
        "example": "She contemplated moving to a new city.",
    },
    {
        "word": "contrary",
        "meaning": "trái ngược",
        "example": "Contrary to popular belief, the earth is not flat.",
    },
    {
        "word": "convenient",
        "meaning": "thuận tiện",
        "example": "The new store is in a convenient location.",
    },
    {
        "word": "conviction",
        "meaning": "niềm tin",
        "example": "She spoke with conviction about her ideas.",
    },
    {
        "word": "credible",
        "meaning": "đáng tin",
        "example": "The witness provided a credible account of the event.",
    },
    {
        "word": "cultivate",
        "meaning": "vun đắp",
        "example": "She cultivated a strong relationship with her clients.",
    },
    {
        "word": "cumulative",
        "meaning": "tích lũy",
        "example": "The cumulative effect of stress can be harmful.",
    },
    {
        "word": "decisive",
        "meaning": "quyết đoán",
        "example": "Her decisive action saved the project.",
    },
    {
        "word": "deduce",
        "meaning": "suy ra",
        "example": "From the evidence, we deduced who was responsible.",
    },
    {
        "word": "deficient",
        "meaning": "thiếu hụt",
        "example": "The report was deficient in key details.",
    },
    {
        "word": "deliberate",
        "meaning": "cố ý",
        "example": "The mistake was deliberate, not accidental.",
    },
    {
        "word": "dense",
        "meaning": "dày đặc",
        "example": "The forest was too dense to walk through easily.",
    },
    {
        "word": "depict",
        "meaning": "miêu tả",
        "example": "The painting depicts a peaceful countryside scene.",
    },
    {
        "word": "deploy",
        "meaning": "triển khai",
        "example": "The company deployed new software across all branches.",
    },
    {
        "word": "deter",
        "meaning": "ngăn cản",
        "example": "High costs deterred them from buying the house.",
    },
    {
        "word": "differentiate",
        "meaning": "phân biệt",
        "example": "It’s important to differentiate between facts and opinions.",
    },
    {
        "word": "diligent",
        "meaning": "chăm chỉ",
        "example": "Her diligent work earned her a promotion.",
    },
    {
        "word": "disclose",
        "meaning": "tiết lộ",
        "example": "He refused to disclose the details of the deal.",
    },
    {
        "word": "discrete",
        "meaning": "riêng biệt",
        "example": "The project was divided into discrete tasks.",
    },
    {
        "word": "disperse",
        "meaning": "phân tán",
        "example": "The crowd dispersed after the event ended.",
    },
    {
        "word": "disrupt",
        "meaning": "làm gián đoạn",
        "example": "The storm disrupted the city’s power supply.",
    },
    {
        "word": "dissolve",
        "meaning": "tan rã",
        "example": "The sugar dissolved quickly in the hot water.",
    },
    {
        "word": "diverge",
        "meaning": "phân kỳ",
        "example": "Their opinions diverged on the best approach.",
    },
    {
        "word": "diverse",
        "meaning": "đa dạng",
        "example": "The team consists of people from diverse backgrounds.",
    },
    {
        "word": "drastic",
        "meaning": "mạnh mẽ",
        "example": "The company took drastic measures to cut costs.",
    },
    {
        "word": "dwell",
        "meaning": "suy nghĩ nhiều",
        "example": "Don’t dwell on your mistakes; learn from them.",
    },
    {
        "word": "elaborate",
        "meaning": "chi tiết",
        "example": "She gave an elaborate explanation of her theory.",
    },
    {
        "word": "elicit",
        "meaning": "gợi ra",
        "example": "The question elicited a strong response from the audience.",
    },
    {
        "word": "endorse",
        "meaning": "ủng hộ",
        "example": "The celebrity endorsed the new product.",
    },
    {
        "word": "enforce",
        "meaning": "thực thi",
        "example": "The police enforced the new traffic laws.",
    },
    {
        "word": "enrich",
        "meaning": "làm giàu",
        "example": "Reading books can enrich your knowledge.",
    },
    {
        "word": "enroll",
        "meaning": "đăng ký",
        "example": "She enrolled in an online course to learn coding.",
    },
    {
        "word": "entail",
        "meaning": "kéo theo",
        "example": "The job entails working long hours.",
    },
    {
        "word": "enthusiastic",
        "meaning": "nhiệt tình",
        "example": "The students were enthusiastic about the field trip.",
    },
    {
        "word": "evoke",
        "meaning": "gợi lên",
        "example": "The music evoked memories of her childhood.",
    },
    {
        "word": "exaggerate",
        "meaning": "phóng đại",
        "example": "He tends to exaggerate his achievements.",
    },
    {
        "word": "exempt",
        "meaning": "miễn trừ",
        "example": "Students are exempt from the exam if they pass the course.",
    },
    {
        "word": "exhaustive",
        "meaning": "toàn diện",
        "example": "The report provided an exhaustive analysis of the data.",
    },
    {
        "word": "expand",
        "meaning": "mở rộng",
        "example": "The company plans to expand its operations overseas.",
    },
    {
        "word": "explicit",
        "meaning": "rõ ràng",
        "example": "The instructions were explicit and easy to follow.",
    },
    {
        "word": "exploit",
        "meaning": "bóc lột",
        "example": "The company was criticized for exploiting its workers.",
    },
    {
        "word": "extract",
        "meaning": "trích xuất",
        "example": "They extracted information from the database.",
    },
    {
        "word": "favorable",
        "meaning": "thuận lợi",
        "example": "The weather was favorable for the outdoor event.",
    },
    {
        "word": "feasible",
        "meaning": "khả thi",
        "example": "The proposal is feasible within the budget.",
    },
    {
        "word": "fluctuate",
        "meaning": "biến động",
        "example": "Prices fluctuate depending on demand.",
    },
    {
        "word": "formidable",
        "meaning": "ghê gớm",
        "example": "The team faced a formidable opponent in the final.",
    },
    {
        "word": "fragile",
        "meaning": "mong manh",
        "example": "The glass vase is very fragile, so handle it carefully.",
    },
    {
        "word": "frustrate",
        "meaning": "làm thất vọng",
        "example": "The delays frustrated everyone on the team.",
    },
    {
        "word": "fulfill",
        "meaning": "hoàn thành",
        "example": "She fulfilled her promise to help the community.",
    },
    {
        "word": "gauge",
        "meaning": "đo lường",
        "example": "The survey helped gauge public opinion.",
    },
    {
        "word": "generate",
        "meaning": "sản sinh",
        "example": "The new policy generated a lot of debate.",
    },
    {
        "word": "gratitude",
        "meaning": "lòng biết ơn",
        "example": "She expressed her gratitude for their support.",
    },
    {
        "word": "harmonious",
        "meaning": "hòa hợp",
        "example": "They have a harmonious working relationship.",
    },
    {
        "word": "hinder",
        "meaning": "cản trở",
        "example": "Bad weather hindered the construction work.",
    },
    {
        "word": "hypothetical",
        "meaning": "giả định",
        "example": "The discussion was based on a hypothetical scenario.",
    },
    {
        "word": "illuminate",
        "meaning": "chiếu sáng",
        "example": "The lamp illuminated the entire room.",
    },
    {
        "word": "imminent",
        "meaning": "sắp xảy ra",
        "example": "The storm is imminent, so stay indoors.",
    },
    {
        "word": "impartial",
        "meaning": "không thiên vị",
        "example": "The referee remained impartial during the match.",
    },
    {
        "word": "implement",
        "meaning": "triển khai",
        "example": "The school implemented a new curriculum.",
    },
    {
        "word": "incline",
        "meaning": "nghiêng",
        "example": "I’m inclined to agree with your suggestion.",
    },
    {
        "word": "inconsistent",
        "meaning": "không nhất quán",
        "example": "His answers were inconsistent with the facts.",
    },
    {
        "word": "indispensable",
        "meaning": "không thể thiếu",
        "example": "Her expertise is indispensable to the team.",
    },
    {
        "word": "inefficient",
        "meaning": "không hiệu quả",
        "example": "The old system was inefficient and slow.",
    },
    {
        "word": "inherent",
        "meaning": "vốn có",
        "example": "There are inherent risks in any investment.",
    },
    {
        "word": "initiate",
        "meaning": "bắt đầu",
        "example": "They initiated a campaign to raise awareness.",
    },
    {
        "word": "inquiry",
        "meaning": "cuộc điều tra",
        "example": "The police launched an inquiry into the incident.",
    },
    {
        "word": "inspire",
        "meaning": "truyền cảm hứng",
        "example": "Her success story inspired many young people.",
    },
    {
        "word": "instinct",
        "meaning": "bản năng",
        "example": "Her instinct told her something was wrong.",
    },
    {
        "word": "integral",
        "meaning": "thiết yếu",
        "example": "Teamwork is integral to the project’s success.",
    },
    {
        "word": "interact",
        "meaning": "tương tác",
        "example": "Students interact with the teacher during lessons.",
    },
    {
        "word": "intervene",
        "meaning": "can thiệp",
        "example": "The manager intervened to resolve the conflict.",
    },
    {
        "word": "intricate",
        "meaning": "phức tạp",
        "example": "The design features intricate patterns.",
    },
    {
        "word": "intrigue",
        "meaning": "gây tò mò",
        "example": "The mystery novel intrigued her from the start.",
    },
    {
        "word": "intuitive",
        "meaning": "trực giác",
        "example": "The app is intuitive and easy to use.",
    },
    {
        "word": "invaluable",
        "meaning": "vô giá",
        "example": "Her advice was invaluable during the crisis.",
    },
    {
        "word": "invoke",
        "meaning": "kêu gọi",
        "example": "The speaker invoked the audience’s emotions.",
    },
    {
        "word": "isolate",
        "meaning": "cô lập",
        "example": "They isolated the problem to a single component.",
    },
    {
        "word": "legitimate",
        "meaning": "hợp pháp",
        "example": "The company operates a legitimate business.",
    },
    {
        "word": "leverage",
        "meaning": "tận dụng",
        "example": "They leveraged their resources to gain an advantage.",
    },
    {
        "word": "lucrative",
        "meaning": "sinh lợi",
        "example": "The deal proved to be highly lucrative.",
    },
    {
        "word": "magnitude",
        "meaning": "độ lớn",
        "example": "The magnitude of the problem was overwhelming.",
    },
    {
        "word": "mandatory",
        "meaning": "bắt buộc",
        "example": "Attendance at the meeting is mandatory.",
    },
    {
        "word": "mediate",
        "meaning": "hòa giải",
        "example": "She mediated the dispute between the two parties.",
    },
    {
        "word": "meticulous",
        "meaning": "cẩn thận",
        "example": "He is meticulous in checking every detail.",
    },
    {
        "word": "minimal",
        "meaning": "tối thiểu",
        "example": "The changes had a minimal impact on the results.",
    },
    {
        "word": "mobilize",
        "meaning": "huy động",
        "example": "The government mobilized resources to aid the disaster area.",
    },
    {
        "word": "modify",
        "meaning": "sửa đổi",
        "example": "They modified the design to improve efficiency.",
    },
    {
        "word": "navigate",
        "meaning": "điều hướng",
        "example": "She navigated the complex system with ease.",
    },
    {
        "word": "neglect",
        "meaning": "bỏ bê",
        "example": "Don’t neglect your health for work.",
    },
    {
        "word": "notable",
        "meaning": "đáng chú ý",
        "example": "The event attracted many notable guests.",
    },
    {
        "word": "notorious",
        "meaning": "tai tiếng",
        "example": "The city is notorious for its traffic jams.",
    },
    {
        "word": "nurture",
        "meaning": "nuôi dưỡng",
        "example": "Parents should nurture their children’s talents.",
    },
    {
        "word": "obscure",
        "meaning": "mờ nhạt",
        "example": "The meaning of the poem was obscure to many readers.",
    },
    {
        "word": "optimism",
        "meaning": "lạc quan",
        "example": "Her optimism inspired the whole team.",
    },
    {
        "word": "orchestrate",
        "meaning": "sắp xếp",
        "example": "She orchestrated the event with great success.",
    },
    {
        "word": "outweigh",
        "meaning": "vượt trội",
        "example": "The benefits of the plan outweigh the risks.",
    },
    {
        "word": "oversee",
        "meaning": "giám sát",
        "example": "He oversees the production process at the factory.",
    },
    {
        "word": "paradox",
        "meaning": "nghịch lý",
        "example": "The paradox is that less work led to better results.",
    },
    {
        "word": "penetrate",
        "meaning": "thâm nhập",
        "example": "The company penetrated the international market.",
    },
    {
        "word": "persevere",
        "meaning": "kiên trì",
        "example": "She persevered despite many challenges.",
    },
    {
        "word": "plausible",
        "meaning": "hợp lý",
        "example": "His explanation seemed plausible at the time.",
    },
    {
        "word": "precaution",
        "meaning": "biện pháp phòng ngừa",
        "example": "They took precautions to avoid accidents.",
    },
    {
        "word": "predominant",
        "meaning": "chiếm ưu thế",
        "example": "English is the predominant language in global business.",
    },
    {
        "word": "prestigious",
        "meaning": "danh giá",
        "example": "She graduated from a prestigious university.",
    },
    {
        "word": "proactive",
        "meaning": "chủ động",
        "example": "Being proactive can prevent future problems.",
    },
    {
        "word": "proficient",
        "meaning": "thành thạo",
        "example": "She is proficient in three languages.",
    },
    {
        "word": "prolong",
        "meaning": "kéo dài",
        "example": "The delay prolonged the meeting by an hour.",
    },
    {
        "word": "prosper",
        "meaning": "thịnh vượng",
        "example": "The town has prospered due to tourism.",
    },
    {
        "word": "provoke",
        "meaning": "khiêu khích",
        "example": "His comments provoked a strong reaction.",
    },
    {
        "word": "prudent",
        "meaning": "thận trọng",
        "example": "It’s prudent to save money for emergencies.",
    },
    {
        "word": "quantify",
        "meaning": "định lượng",
        "example": "It’s difficult to quantify the impact of the campaign.",
    },
    {
        "word": "radical",
        "meaning": "cấp tiến",
        "example": "The government introduced radical changes to the system.",
    },
    {
        "word": "reassure",
        "meaning": "trấn an",
        "example": "The doctor reassured the patient about the treatment.",
    },
    {
        "word": "reconcile",
        "meaning": "hòa giải",
        "example": "They reconciled their differences and worked together.",
    },
    {
        "word": "refine",
        "meaning": "tinh chỉnh",
        "example": "The team refined the design to make it more efficient.",
    },
    {
        "word": "reinforce",
        "meaning": "củng cố",
        "example": "The new data reinforced their hypothesis.",
    },
    {
        "word": "relocate",
        "meaning": "di dời",
        "example": "The company relocated its office to a new city.",
    },
    {
        "word": "renowned",
        "meaning": "nổi tiếng",
        "example": "The chef is renowned for his creative dishes.",
    },
    {
        "word": "repercussion",
        "meaning": "hậu quả",
        "example": "The decision had serious repercussions for the company.",
    },
    {
        "word": "resilient",
        "meaning": "kiên cường",
        "example": "She is resilient and bounces back from setbacks.",
    },
    {
        "word": "restrain",
        "meaning": "kiềm chế",
        "example": "He restrained his anger during the argument.",
    },
    {
        "word": "retrieve",
        "meaning": "lấy lại",
        "example": "She retrieved her files from the cloud storage.",
    },
    {
        "word": "rigorous",
        "meaning": "nghiêm ngặt",
        "example": "The training program is rigorous but effective.",
    },
    {
        "word": "scrutinize",
        "meaning": "xem xét kỹ lưỡng",
        "example": "The committee scrutinized the proposal carefully.",
    },
    {
        "word": "segregate",
        "meaning": "phân chia",
        "example": "They segregated the waste for recycling.",
    },
    {
        "word": "speculative",
        "meaning": "suy đoán",
        "example": "The article was based on speculative assumptions.",
    },
    {
        "word": "stagnant",
        "meaning": "đình trệ",
        "example": "The economy has been stagnant for months.",
    },
    {
        "word": "strategic",
        "meaning": "chiến lược",
        "example": "The company made a strategic decision to expand.",
    },
    {
        "word": "subordinate",
        "meaning": "cấp dưới",
        "example": "The manager delegated tasks to her subordinates.",
    },
    {
        "word": "suppress",
        "meaning": "đàn áp",
        "example": "She suppressed her emotions during the meeting.",
    },
    {
        "word": "sustain",
        "meaning": "duy trì",
        "example": "The company cannot sustain such high costs.",
    },
    {
        "word": "tangible",
        "meaning": "hữu hình",
        "example": "The project delivered tangible benefits to the community.",
    },
    {
        "word": "tedious",
        "meaning": "nhàm chán",
        "example": "The task was tedious but necessary.",
    },
    {
        "word": "tentative",
        "meaning": "tạm thời",
        "example": "They made a tentative agreement to meet next week.",
    },
    {
        "word": "thorough",
        "meaning": "kỹ lưỡng",
        "example": "The investigation was thorough and detailed.",
    },
    {
        "word": "transcend",
        "meaning": "vượt qua",
        "example": "Her work transcends traditional boundaries.",
    },
    {
        "word": "unprecedented",
        "meaning": "chưa từng có",
        "example": "The event attracted an unprecedented number of visitors.",
    },
    {
        "word": "uphold",
        "meaning": "duy trì",
        "example": "The court upheld the original decision.",
    },
    {
        "word": "validate",
        "meaning": "xác nhận",
        "example": "The experiment validated their theory.",
    },
    {
        "word": "variable",
        "meaning": "biến đổi",
        "example": "Weather conditions are highly variable in this region.",
    },
    {
        "word": "vibrant",
        "meaning": "sôi động",
        "example": "The city has a vibrant cultural scene.",
    },
    {
        "word": "vigilant",
        "meaning": "cảnh giác",
        "example": "Security guards must remain vigilant at all times.",
    },
    {
        "word": "voluntary",
        "meaning": "tự nguyện",
        "example": "She joined the voluntary organization to help others.",
    },
    {
        "word": "warrant",
        "meaning": "bảo đảm",
        "example": "The situation does not warrant such a strong reaction.",
    }
]

def get_sequential_vocabulary_words(count=5):
    """Lấy từ vựng theo thứ tự dựa trên ngày tháng để tránh trùng lặp"""
    try:
        # Sử dụng ngày tháng để tính toán vị trí bắt đầu
        # Mỗi ngày sẽ có một vị trí bắt đầu khác nhau
        now = datetime.now()
        
        # Tính số ngày từ ngày 1/1/2024 để có một chuỗi số ổn định
        start_date = datetime(2024, 1, 1)
        days_since_start = (now - start_date).days
        
        # Tính vị trí bắt đầu dựa trên ngày và giờ
        # Mỗi ngày có 3 lần gửi (6h, 14h, 22h UTC)
        hour = now.hour
        if hour < 10:  # 6h UTC
            session = 0
        elif hour < 18:  # 14h UTC  
            session = 1
        else:  # 22h UTC
            session = 2
            
        # Tính vị trí bắt đầu: mỗi ngày có 3 session, mỗi session 5 từ
        words_per_day = 15  # 3 sessions × 5 words
        current_index = (days_since_start * words_per_day) + (session * count)
        
        # Lấy từ vựng theo thứ tự
        words = []
        total_words = len(VOCABULARY_DATABASE)
        
        for i in range(count):
            # Sử dụng modulo để quay vòng khi hết từ vựng
            word_index = (current_index + i) % total_words
            words.append(VOCABULARY_DATABASE[word_index])
        
        logger.info(f"📚 Đã gửi {count} từ vựng, vị trí bắt đầu: {current_index}, ngày: {now.strftime('%d/%m/%Y')}, giờ: {now.hour}h")
        return words
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy từ vựng theo thứ tự: {e}")
        # Fallback về random nếu có lỗi
        return random.sample(VOCABULARY_DATABASE, min(count, len(VOCABULARY_DATABASE)))

def format_vocabulary_message(words):
    """Định dạng tin nhắn từ vựng cho Telegram"""
    message = "📚 <b>HỌC TỪ VỰNG TIẾNG ANH HÔM NAY</b>\n\n"
    
    for i, word_data in enumerate(words, 1):
        message += f"<b>{i}. {word_data['word'].upper()}</b>\n"
        message += f"📖 <i>{word_data['meaning']}</i>\n"
        message += f"💬 <i>\"{word_data['example']}\"</i>\n\n"
    
    message += f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
    message += "💡 Học 5 từ mỗi bài để cải thiện tiếng Anh!"
    
    return message

def send_vocabulary_lesson():
    """Gửi bài học từ vựng qua Telegram riêng biệt"""
    try:
        import requests
        from vocabulary_config import VOCABULARY_BOT_TOKEN, VOCABULARY_CHAT_ID
        
        # Lấy 5 từ theo thứ tự lần lượt
        words = get_sequential_vocabulary_words(5)
        
        # Tạo tin nhắn
        message = format_vocabulary_message(words)
        
        # Gửi qua Telegram bot riêng
        url = f"https://api.telegram.org/bot{VOCABULARY_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': VOCABULARY_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Đã gửi bài học từ vựng tiếng Anh thành công!")
            return True
        else:
            logger.error(f"❌ Lỗi khi gửi từ vựng: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi gửi bài học từ vựng: {e}")
        return False


def get_vocabulary_progress():
    """Lấy tiến độ học từ vựng dựa trên ngày tháng"""
    try:
        now = datetime.now()
        start_date = datetime(2024, 1, 1)
        days_since_start = (now - start_date).days
        
        # Tính session hiện tại
        hour = now.hour
        if hour < 10:  # 6h UTC
            session = 0
        elif hour < 18:  # 14h UTC  
            session = 1
        else:  # 22h UTC
            session = 2
            
        words_per_day = 15  # 3 sessions × 5 words
        current_index = (days_since_start * words_per_day) + (session * 5)
        
        return {
            'current_index': current_index,
            'last_updated': now.isoformat(),
            'total_words': len(VOCABULARY_DATABASE),
            'words_sent': 5,
            'days_since_start': days_since_start,
            'current_session': session,
            'current_hour': hour
        }
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc tiến độ: {e}")
        return None

def reset_vocabulary_progress():
    """Reset tiến độ học từ vựng về đầu (không cần thiết với logic mới)"""
    logger.info("🔄 Logic mới không cần reset, tiến độ được tính dựa trên ngày tháng")
    return True

if __name__ == "__main__":
    # Test function
    print("🧪 Testing vocabulary learning module...")
    
    # Test getting sequential words
    words = get_sequential_vocabulary_words(3)
    print(f"📚 Sequential words: {[w['word'] for w in words]}")
    
    # Test formatting message
    message = format_vocabulary_message(words)
    print("\n📱 Formatted message:")
    print(message)
    
    # Test progress
    progress = get_vocabulary_progress()
    print(f"\n📊 Vocabulary progress: {progress}")
