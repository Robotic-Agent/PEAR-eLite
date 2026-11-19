export const repo = 'https://github.com/Robotic-Agent/PEAR-eLite'
export const base = import.meta.env.BASE_URL.replace(/\/$/, '')
export const url = (path = '') => `${base}/${path ? `${path}/` : ''}`
