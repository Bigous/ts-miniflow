import { Node } from '../src/node';

test('Should greet with message', () => {
  const greeter = new Node('friend');
  expect(greeter.greet()).toBe('Bonjour, friend!');
});
