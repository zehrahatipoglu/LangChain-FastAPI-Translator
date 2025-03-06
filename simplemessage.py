from platform import system, version  # İşletim sistemi ve versiyon bilgilerini almak için
from click import prompt  # Kullanıcıdan komut satırında giriş almak için
from dotenv import load_dotenv  # .env dosyasındaki API anahtarlarını ve değişkenleri yüklemek için

# LangChain ile ilgili gerekli modülleri içe aktarıyoruz
from langchain.chains.question_answering.map_reduce_prompt import \
    messages  # Soru-cevap için map-reduce prompt şablonları
from langchain.chains.summarize.map_reduce_prompt import \
    prompt_template  # Özetleme işlemi için map-reduce prompt şablonu
from langchain.chains.summarize.refine_prompts import prompt_template  # Özetleme işlemi için refine prompt şablonu
from langchain_openai import ChatOpenAI  # OpenAI'nin dil modeli için LangChain wrapper'ı
from langchain_core.messages import HumanMessage, SystemMessage  # Model ile mesaj alışverişi için temel sınıflar
from langchain_core.output_parsers import StrOutputParser  # Çıktıyı işleyip string olarak almak için
from langchain_core.prompts import ChatPromptTemplate  # Prompt şablonları oluşturmak için
from fastapi import FastAPI  # FastAPI ile web uygulaması oluşturmak için
from langserve import add_routes  # LangChain'in FastAPI ile entegrasyonu için

# .env dosyasını yükleyerek API anahtarlarını ve diğer çevresel değişkenleri kullanabilir hale getiriyoruz
load_dotenv()

# OpenAI'nin GPT-4 modelini belirli bir sıcaklık değeriyle (temperature) başlatıyoruz
# Temperature değeri, modelin rastgeleliğini belirler. 0.1 gibi düşük bir değer modelin daha tutarlı yanıtlar vermesini sağlar.
model = ChatOpenAI(model="gpt-4", temperature=0.1)

# Alternatif bir mesaj formatı ile modelden yanıt alabiliriz (kapalı, alternatif bir kullanım olduğu için şimdilik yorum satırına aldık)
# messages = [
#     SystemMessage(content="Translate the following from English to Spanish"),
#     HumanMessage(content="Hi"),
# ]

# Prompt şablonumuz, modele hangi talimatı verdiğimizi belirler.
# Burada, "Verilen metni belirtilen dile çevir" şeklinde bir sistem mesajı belirtiyoruz.
system_prompt = "Translate the following into {language}"

# Kullanıcıdan gelecek girdiye göre dinamik bir prompt şablonu oluşturuyoruz.
# "system" mesajı modeli yönlendirirken, "user" mesajı kullanıcı girdisini temsil eder.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Modelin genel görevini belirleyen sistem mesajı
        ("user", "{text}")  # Kullanıcının çevirmek istediği metni modele gönderir
    ]
)

# Model çıktısını işleyip string olarak almak için bir parser kullanıyoruz
parser = StrOutputParser()

# Alternatif bir yöntemle modeli doğrudan çağırarak yanıt alabiliriz (şimdilik yorum satırına aldık)
# response = model.invoke(messages)

# LangChain'de "chain" kullanarak, farklı işlemleri birbirine bağlayarak akış oluşturabiliyoruz.
# Burada önce prompt şablonunu oluşturup, modele gönderiyoruz, ardından çıktıyı parser ile işliyoruz.
chain = prompt_template | model | parser

# FastAPI uygulamasını başlatıyoruz
app = FastAPI(
    title="Translator App",  # API için bir başlık belirtiyoruz
    description="Translation Chat Bot"  # API hakkında kısa bir açıklama
)

# LangChain'in FastAPI ile entegrasyonunu sağlıyoruz
# Yani, "chain" yapımızı bir API endpoint olarak ekliyoruz.
add_routes(
    app,  # FastAPI uygulaması
    chain,  # LangChain "chain" nesnesi
    path="/chain"  # API'nin hangi URL yolundan erişileceğini belirtiyoruz
)

# Eğer bu dosya doğrudan çalıştırılıyorsa, FastAPI sunucusunu başlatıyoruz.
if __name__ == "__main__":
    # Alternatif olarak farklı şekillerde modelin çıktısını almak mümkün:
    # 1. Standart OpenAI API çağrısı
    # response = model.invoke(messages)
    # print(response.content)

    # 2. Parser kullanarak çıktıyı düzenlemek
    # print(parser.invoke(response))

    # 3. Zincirleme yapı ile prompt şablonu + model + parser kullanarak yanıt almak
    # print(chain.invoke(messages))

    # 4. Örnek bir çeviri çağrısı
    # print(chain.invoke({"language": "italian", "text": "Hello World"}))

    # FastAPI sunucusunu başlatıyoruz
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
