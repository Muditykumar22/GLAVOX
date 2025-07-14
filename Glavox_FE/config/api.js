// Get the local IP address from your computer
const getLocalIP = () => {
  // You can replace this with your computer's IP address
  return '44.217.120.1'; // Replace with your actual local IP address
};

const ENV = {
  dev: {
    API_URL: `http://${getLocalIP()}:5000/api`,
    SOCKET_URL: `http://${getLocalIP()}:5000`,
    FLASK_API_URL: `http://${getLocalIP()}:5001`
  },
  prod: {
    API_URL: 'https://your-production-api.com/api', // Replace with your production API URL
    SOCKET_URL: 'https://your-production-api.com',
    FLASK_API_URL: 'https://your-production-flask-api.com'
  }
};

const getEnvVars = () => {
  if (__DEV__) {
    return ENV.dev;
  }
  return ENV.prod;
};

export default getEnvVars(); 